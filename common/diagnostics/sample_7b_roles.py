"""Record the fixed GPU on one node during the paired 7B diagnostics."""

import argparse
import json
import os
from pathlib import Path
from statistics import fmean
import subprocess
import time


PHASES = (("stage-timing", ("B", "C70", "C50")), ("tail-p99", ("A", "B", "C70", "C50")))


def live(record):
    proc = Path("/proc") / str(record["pid"])
    try:
        return (proc.stat().st_uid == os.getuid()
                and (proc / "cmdline").read_bytes().split(b"\0") == [*map(os.fsencode, record["command"]), b""])
    except FileNotFoundError:
        return False


def sample(gpu_uuid):
    text = subprocess.check_output([
        "nvidia-smi", "-i", gpu_uuid, "--query-gpu=utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=5)
    utilization, memory = map(int, text.strip().split(","))
    if not 0 <= utilization <= 100 or memory < 0:
        raise ValueError("Invalid GPU utilization or memory sample")
    return dict(utilization_pct=utilization, used_mib=memory)


def summarize_window(run, stem, requests):
    path = run / f"{stem}.gpu-samples.jsonl"
    if not path.is_file():
        return None
    gpu_uuid = json.loads((run / f"{stem}.launch.json").read_text(encoding="utf-8"))["gpu_uuid"]
    origin = requests[0]["wall_origin_ns"] / 1e9
    sent = [row for row in requests if row["send_offset_s"] is not None]
    begin = origin + min(row["send_offset_s"] for row in sent)
    end = origin + max(row["send_offset_s"] + row["duration_s"] for row in sent)
    entries, errors = [], 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            row = json.loads(line)
            if row["gpu_uuid"] != gpu_uuid:
                raise ValueError(f"{path}: sampled GPU differs from the launch receipt")
            if begin <= row["epoch_s"] <= end:
                if "error" in row:
                    errors += 1
                else:
                    entries.append(row)
    return dict(samples=len(entries), sample_errors=errors, window_s=end - begin,
                first_sample_offset_s=entries[0]["epoch_s"] - begin if entries else None,
                last_sample_offset_s=entries[-1]["epoch_s"] - begin if entries else None,
                mean_utilization_pct=fmean(row["utilization_pct"] for row in entries) if entries else None,
                peak_used_mib=max((row["used_mib"] for row in entries), default=None))


def observe(run, node, gpu_uuid, deadline):
    for phase, groups in PHASES:
        failures = [run / f"{phase}-{peer}.failure.json" for peer in ("qjy000", "qjy001")]
        if phase == "tail-p99":
            failures.append(run / f"tail-p99-{node}.deferred.json")
        for group in groups:
            stem = f"{group}-{node}-{phase}"
            launch = run / f"{stem}.launch.json"
            release = run / f"{stem}.release.json"
            record = None
            while time.time() < deadline and not any(path.exists() for path in failures):
                if release.exists():
                    break
                if launch.is_file():
                    try:
                        record = json.loads(launch.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        time.sleep(1)
                        continue
                    break
                time.sleep(2)
            if time.time() >= deadline or any(path.exists() for path in failures):
                return
            if record is None:
                continue
            if record["gpu_uuid"] != gpu_uuid:
                raise ValueError(f"{launch}: GPU does not match the requested allocation")
            output = run / f"{stem}.gpu-samples.jsonl"
            with output.open("x", encoding="utf-8", buffering=1) as stream:
                while time.time() < deadline and not release.exists() and live(record):
                    started, tick = time.time(), time.monotonic()
                    try:
                        result = sample(gpu_uuid)
                    except (OSError, subprocess.SubprocessError, ValueError) as error:
                        result = dict(error=f"{type(error).__name__}: {error}")
                    epoch_s = (started + time.time()) / 2
                    print(json.dumps(dict(epoch_s=epoch_s, gpu_uuid=gpu_uuid, **result)), file=stream)
                    time.sleep(max(0, 1 - (time.monotonic() - tick)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--node", choices=("qjy000", "qjy001"), required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--deadline-epoch-s", type=float, required=True)
    args = parser.parse_args()
    observe(args.run_dir, args.node, args.gpu_uuid, args.deadline_epoch_s)


if __name__ == "__main__":
    main()
