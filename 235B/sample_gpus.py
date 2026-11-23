"""Sample GPUs from the managed 235B launch receipts without touching the replay."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

from summarize import ENCODER_PORTS, MODEL_REVISION, PD_PORT, SGLANG_REVISION


def sample(gpus):
    output = subprocess.check_output([
        "nvidia-smi", "-i", ",".join(gpus), "--query-gpu=uuid,utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=5)
    rows = {
        uuid: dict(utilization_pct=int(utilization), used_mib=int(memory))
        for uuid, utilization, memory in csv.reader(output.splitlines(), skipinitialspace=True)
    }
    if set(rows) != set(gpus) or any(not 0 <= m["utilization_pct"] <= 100 or m["used_mib"] < 0 for m in rows.values()):
        raise ValueError("GPU sample does not match the launch allocation")
    return rows


def observe(run, deadline):
    def finished():
        return time.time() >= deadline or any((run / name).exists() for name in ("scale.failed.json", "scale-summary.json"))

    def wait_for(paths):
        while not finished():
            if all(path.is_file() for path in paths):
                return True
            time.sleep(1)
        return False

    start = run / "scale.started.json"
    if not wait_for([start]):
        return
    session = json.loads(start.read_text(encoding="utf-8"))
    if (session["model_revision"], session["source_commit"]) != (MODEL_REVISION, SGLANG_REVISION):
        raise ValueError("GPU sampler requires the fixed 235B session")

    for count in (4, 5, 6):
        group = f"235B-{count}"
        port_counts = [(PD_PORT, session["tp4"])]
        port_counts.extend((port, [gpu]) for port, gpu in zip(ENCODER_PORTS[:count - 4], session["encoders"]))
        launches = [run / f"{group}-{port}.launch.json" for port, _ in port_counts]
        if not wait_for(launches):
            return
        gpus = []
        for path, (_, expected) in zip(launches, port_counts, strict=True):
            value = json.loads(path.read_text(encoding="utf-8"))
            if (value["gpu_uuids"], value["model_revision"], value["source_commit"]) != (
                expected, MODEL_REVISION, SGLANG_REVISION,
            ):
                raise ValueError(f"{path}: launch differs from recorded devices or model")
            gpus.extend(expected)
        if len(gpus) != count or len(set(gpus)) != count:
            raise ValueError(f"{group}: GPU allocation has invalid cardinality")
        releases = [run / f"{group}-{port}.release.json" for port, _ in port_counts]
        with (run / f"{group}.gpu-samples.jsonl").open("x", encoding="utf-8", buffering=1) as stream:
            while not finished() and not any(path.exists() for path in releases):
                began = time.time()
                tick = time.monotonic()
                try:
                    result = dict(gpus=sample(gpus))
                except (OSError, subprocess.SubprocessError, ValueError) as error:
                    result = dict(error=f"{type(error).__name__}: {error}")
                timestamp = (began + time.time()) / 2
                print(json.dumps(dict(epoch_s=timestamp, **result)), file=stream)
                time.sleep(max(0, 1 - (time.monotonic() - tick)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--deadline-epoch-s", type=float, required=True)
    args = parser.parse_args()
    observe(args.run_dir, args.deadline_epoch_s)


if __name__ == "__main__":
    main()
