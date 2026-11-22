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


def frozen_requests(path, case):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != case["trace_sha256"]:
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
            throughput = metrics["completed_requests_per_s"]
            runs.append(dict(
                case=name, gpu_count=gpus, tp4=tp4, encoders=encoders[:gpus-4],
                completion_rate=metrics["completed"] / count,
                mean_ttft_ms=fmean(row["content_chunks"][0][0] * 1000 for row in completed) if completed else None,
                mean_tpot_ms=fmean(actual_tpot) if actual_tpot else None,
                completed_requests_per_s_per_gpu=throughput / gpus if throughput is not None else None,
                **metrics,
            ))
    return {"model_revision": MODEL_REVISION, "cases": cases, "runs": runs}


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
