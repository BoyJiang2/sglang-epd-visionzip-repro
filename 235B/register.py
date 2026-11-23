"""Register the two fixed scale cases and publish the audited 7B freeze."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


MODEL_REVISION = "7fbcd8c9e2ad011808ed8a57d64c462605c3e381"
POINTS = ((1, 0.5), (8, 1))
GROUPS = ("B", "C70", "C50")
NODES = ("qjy000", "qjy001")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(file):
    value = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_7b(run):
    root = run / "analysis-7b"
    receipt = root / "analysis-receipt.json"
    artifacts = read_json(receipt)["artifacts"]
    required = {
        "raw-evidence.json", "performance.json", "performance.png", "interference.json",
        "interference.png", "muir-traffic.json", "quality-traffic.json", "quality-traffic.png",
        "probe-transport.json", "stage-timing.json",
    }
    if not required <= artifacts.keys() or len(artifacts) != 23:
        raise ValueError("The 7B analysis receipt does not cover every registered product")
    for name, entry in artifacts.items():
        if Path(name).name != name:
            raise ValueError(f"Invalid analysis artifact name {name}")
        file = root / name
        if file.stat().st_size != entry["bytes"] or digest(file) != entry["sha256"]:
            raise ValueError(f"Analysis artifact changed after generation: {name}")
    audit = read_json(root / "raw-evidence.json")
    if (audit["raw_evidence_ready"] is not True or len(audit["evidence"]) != 156
            or any(item["state"] != "verified" for item in audit["evidence"].values())):
        raise ValueError("The complete 7B raw evidence audit must pass")
    for group in GROUPS:
        for node in NODES:
            stem = f"{group}-{node}-stage-timing"
            launch = read_json(run / f"{stem}.launch.json")
            released = read_json(run / f"{stem}.release.json")
            fields = ("pid", "gpu_uuid", "port")
            if (any(launch[key] != released[key] for key in fields)
                    or released["epoch_s"] < launch["start_epoch_s"]
                    or released["remaining_mib"] >= 150):
                raise ValueError(f"No matching 7B GPU release for {stem}")
    return digest(receipt)


def verified_trace(run, name, images, expected):
    source = run / f"{name}.trace.jsonl"
    with (run / f"{name}.A.jsonl").open(encoding="utf-8") as stream:
        rows = list(map(json.loads, stream))
    paired = {
        row["request_id"]: (row["planned_offset_s"], row["image_digests"], row["expected_output_tokens"])
        for row in rows
    }
    value = hashlib.sha256()
    seen = set()
    with source.open("rb") as stream:
        for line in stream:
            value.update(line)
            row = json.loads(line)
            rid = row["id"]
            if (rid in seen or len(row["images"]) != images or len(row["image_digests"]) != images
                    or paired.get(rid) != (row["at"], row["image_digests"], row["output_tokens"])):
                raise ValueError(f"{source.name}: differs from 7B replay at {rid}")
            seen.add(rid)
    if len(rows) != expected or len(paired) != expected or seen != paired.keys():
        raise ValueError(f"{source.name}: expected {expected} paired requests")
    return source, value.hexdigest()


def publish_json(path, value):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as draft:
        json.dump(value, draft, indent=2)
        draft.write("\n")
        draft.flush()
        os.fsync(draft.fileno())
    try:
        os.link(draft.name, path)
    except FileExistsError:
        if read_json(path) != value:
            raise ValueError(f"Conflicting existing registration at {path}")
    finally:
        Path(draft.name).unlink()


def register(run):
    analysis_hash = verify_7b(run)
    root = run / "235b"
    placement = read_json(root / "gpu-placement.preflight.json")
    if placement["model_revision"] != MODEL_REVISION:
        raise ValueError("235B model revision does not match the registered FP8 checkpoint")
    tp4 = [item["gpu_uuid"] for item in placement["tp4"]]
    encoders = [item["gpu_uuid"] for item in placement["encoders"]]
    if ([item["index"] for item in placement["tp4"]] != [4, 5, 6, 7]
            or [item["index"] for item in placement["encoders"]] != [1, 2]
            or len(set(tp4 + encoders)) != 6
            or any(not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", uuid)
                   for uuid in tp4 + encoders)):
        raise ValueError("235B requires the profiled TP4 and independent encoder cards")
    workload = read_json(run / "formal-design.json")
    seed = min(workload["seeds"])
    expected = workload["requests_per_cell"]
    if seed != 23011 or expected != 128 or any(qps not in workload["rates_qps"][str(images)] for images, qps in POINTS):
        raise ValueError("The preselected 235B cases do not match the frozen 7B plan")
    cases, links = [], []
    for images, qps in POINTS:
        name = f"formal-{images}img-{qps:g}qps-seed{seed}"
        source, sha256 = verified_trace(run, name, images, expected)
        cases.append(dict(name=name, requests=expected, trace_sha256=sha256))
        links.append(source)
    for source in links:
        target = root / source.name
        try:
            target.symlink_to(Path("..") / source.name)
        except FileExistsError:
            if not target.is_symlink() or target.resolve() != source.resolve():
                raise ValueError(f"Conflicting existing scale trace {target}")
    design = root / "scale-design.json"
    publish_json(design, dict(model_revision=MODEL_REVISION, tp4=tp4, encoders=encoders, cases=cases))
    publish_json(run / "7b-frozen.json", dict(frozen=True, analysis_receipt_sha256=analysis_hash,
                                             scale_design_sha256=digest(design)))
    return design


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(register(args.run_dir))
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"235B registration stopped: {error}\n")


if __name__ == "__main__":
    main()
