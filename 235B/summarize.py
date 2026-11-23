"""Summarize paired 235B runs on 4, 5, and 6 H20 GPUs."""

import argparse
import hashlib
import json
from pathlib import Path
import re
from statistics import fmean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from epd_replay import summarize_run
from report_performance import fingerprint, load_cell


MODEL_REVISION = "7fbcd8c9e2ad011808ed8a57d64c462605c3e381"
SGLANG_REVISION = "8396d60a14ef03b23d2cdaa5c64adb18eba7c703"
PD_PORT, ENCODER_PORTS, BOOTSTRAP_PORT = 31230, (31231, 31232), 8997


def managed_placements(run_dir, tp4, encoders):
    started = run_dir / "scale.started.json"
    if not started.exists():
        return {}
    session = json.loads(started.read_text(encoding="utf-8"))
    if (session["model_revision"], session["source_commit"], session["tp4"], session["encoders"]) != (
        MODEL_REVISION, SGLANG_REVISION, tp4, encoders,
    ):
        raise ValueError("Scale session differs from the registered model or GPU placement")
    placements = {}
    for gpu_count in (4, 5, 6):
        group = f"235B-{gpu_count}"
        active_encoders = encoders[:gpu_count - 4]
        encoder_ports = ENCODER_PORTS[:len(active_encoders)]
        roles = [(PD_PORT, tp4)]
        roles.extend((port, [gpu]) for port, gpu in zip(encoder_ports, active_encoders))
        services = []
        for port, expected in roles:
            stem = f"{group}-{port}"
            launch = json.loads((run_dir / f"{stem}.launch.json").read_text(encoding="utf-8"))
            release = json.loads((run_dir / f"{stem}.release.json").read_text(encoding="utf-8"))
            command = launch["command"]
            encoder = port != PD_PORT
            ports = [port, BOOTSTRAP_PORT] if not encoder and encoder_ports else [port]
            if (launch["gpu_uuids"] != expected or launch["ports"] != ports
                    or launch["model_revision"] != MODEL_REVISION or launch["source_commit"] != session["source_commit"]
                    or any(launch[key] != release[key] for key in ("pid", "gpu_uuids", "ports"))
                    or release["idle"] is not True or not session["epoch_s"] <= launch["epoch_s"] < release["epoch_s"]
                    or "--port" not in command or command[command.index("--port") + 1:][:1] != [str(port)]
                    or "--tp-size" not in command or command[command.index("--tp-size") + 1:][:1] != [str(len(expected))]
                    or ("--encoder-only" in command) != encoder
                    or ("--language-only" in command) != (not encoder and bool(encoder_ports))):
                raise ValueError(f"{stem}: launch and release do not establish the registered role and devices")
            if not encoder and encoder_ports and (
                "--encoder-urls" not in command
                or command[command.index("--encoder-urls") + 1:] != [f"http://127.0.0.1:{p}" for p in encoder_ports]
            ):
                raise ValueError(f"{stem}: PD routes differ from the registered encoders")
            services.append(dict(port=port, gpu_uuids=launch["gpu_uuids"], pid=launch["pid"],
                                 launch_epoch_s=launch["epoch_s"], release_epoch_s=release["epoch_s"]))
        placements[group] = services
    return placements


def gpu_activity(path, services, start, finish):
    if not path.is_file():
        return None
    roles = {"colocated" if len(services) == 1 else "pd": services[0]["gpu_uuids"]}
    if len(services) > 1:
        roles["encoder"] = [gpu for service in services[1:] for gpu in service["gpu_uuids"]]
    devices = {gpu for gpus in roles.values() for gpu in gpus}
    snapshots, errors = [], 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            row = json.loads(line)
            if not start <= row["epoch_s"] <= finish:
                continue
            if "error" in row:
                errors += 1
                continue
            if set(row["gpus"]) != devices:
                raise ValueError(f"{path}: sampled GPUs differ from the verified service allocation")
            snapshots.append(row)
    return dict(
        samples=len(snapshots), sample_errors=errors,
        first_sample_offset_s=snapshots[0]["epoch_s"] - start if snapshots else None,
        last_sample_offset_s=snapshots[-1]["epoch_s"] - start if snapshots else None,
        roles={
            role: dict(gpu_count=len(gpus),
                       mean_utilization_pct=fmean(row["gpus"][gpu]["utilization_pct"] for row in snapshots for gpu in gpus),
                       peak_used_mib_per_gpu=max(row["gpus"][gpu]["used_mib"] for row in snapshots for gpu in gpus))
            for role, gpus in roles.items()
        } if snapshots else {},
    )


def frozen_requests(path, case):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != case["trace_sha256"]:
        raise ValueError(f"{path}: trace differs from the registered SHA-256")
    requests = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in requests:
                raise ValueError(f"{path}: duplicate request {row['id']}")
            if not row["images"] or len(row["images"]) != len(row["image_digests"]):
                raise ValueError(f"{path}: missing image provenance for {row['id']}")
            requests[row["id"]] = (row["at"], len(row["images"]), row["image_digests"], row["output_tokens"])
    if len(requests) != case["requests"]:
        raise ValueError(f"{path}: request count differs from the registered denominator")
    return requests


def report(run_dir, plan):
    if plan["model_revision"] != MODEL_REVISION:
        raise ValueError("Scale report requires the pinned 235B FP8 checkpoint")
    tp4, encoders = plan["tp4"], plan["encoders"]
    if len(tp4) != 4 or len(encoders) != 2 or len({gpu.lower() for gpu in tp4 + encoders}) != 6:
        raise ValueError("Register four distinct TP4 GPUs and two separate encoder GPUs")
    if any(not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", gpu)
           for gpu in tp4 + encoders):
        raise ValueError("Use GPU UUIDs from the launch receipts")
    cases = plan["cases"]
    if not 2 <= len(cases) <= 3 or len({case["name"] for case in cases}) != len(cases):
        raise ValueError("Register two or three unique load points")

    placements = managed_placements(run_dir, tp4, encoders)
    runs = []
    for case in cases:
        name, count = case["name"], case["requests"]
        if not re.fullmatch(r"[a-zA-Z0-9._-]+", name) or count <= 0:
            raise ValueError(f"Invalid registered load point: {name}")
        expected = frozen_requests(run_dir / f"{name}.trace.jsonl", case)
        for gpus in (4, 5, 6):
            group = f"235B-{gpus}"
            path = run_dir / f"{name}.{group}.jsonl"
            rows = load_cell(path, group, count)
            if {rid: fingerprint(row) for rid, row in rows.items()} != expected:
                raise ValueError(f"{path}: requests differ from the registered trace")
            completed = [row for row in rows.values() if row["status"] == "complete"]
            actual_tpot = [
                (row["duration_s"] - row["content_chunks"][0][0]) * 1000 / (row["completion_tokens"] - 1)
                for row in completed if (row["completion_tokens"] or 0) > 1
            ]
            metrics = summarize_run(path, count)
            activity = None
            if group in placements:
                services = placements[group]
                origin = metrics["wall_origin_ns"] / 1e9
                finish = origin + max(row["send_offset_s"] + row["duration_s"] for row in rows.values())
                if (set(metrics["endpoints"]) != {f"http://127.0.0.1:{PD_PORT}"}
                        or origin < max(s["launch_epoch_s"] for s in services)
                        or finish > min(s["release_epoch_s"] for s in services)):
                    raise ValueError(f"{path}: replay falls outside the verified service endpoints or lifetimes")
                first_send = origin + min(row["send_offset_s"] for row in rows.values())
                activity = gpu_activity(run_dir / f"{group}.gpu-samples.jsonl", services, first_send, finish)
            throughput = metrics["completed_requests_per_s"]
            runs.append(dict(
                case=name, gpu_count=gpus, tp4=tp4, encoders=encoders[:gpus-4],
                completion_rate=metrics["completed"] / count,
                mean_ttft_ms=fmean(row["content_chunks"][0][0] * 1000 for row in completed) if completed else None,
                mean_tpot_ms=fmean(actual_tpot) if actual_tpot else None,
                completed_requests_per_s_per_gpu=throughput / gpus if throughput is not None else None,
                gpu_activity=activity,
                **metrics,
            ))
    return {"model_revision": MODEL_REVISION, "cases": cases, "runtime_placements": placements, "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.run_dir / "scale-design.json").read_text(encoding="utf-8"))
    result = report(args.run_dir, plan)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
