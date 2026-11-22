"""Replay a registered subset of frozen OCRBench and RealWorldQA images."""

import argparse
import json
from pathlib import Path
from random import Random


def extract(run_dir, plan):
    for task, info in plan["tasks"].items():
        path = run_dir / info["source"]
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (info["source_size_bytes"], info["source_mtime_ns"]):
            raise ValueError(f"{path}: frozen file no longer matches the registration")
        samples = info["samples"]
        if len({item["doc_id"] for item in samples}) != len(samples):
            raise ValueError(f"{task}: duplicate sampled questions")
        arrivals = Random(plan["arrival_seed"])
        rows = []
        scheduled = 0.0
        with path.open("rb") as stream:
            for sample in samples:
                stream.seek(sample["file_offset_bytes"])
                row = json.loads(stream.readline())
                image, = row["images"]
                prefix, _, encoded = image.partition(",")
                if (
                    prefix != "data:image/png;base64"
                    or len(encoded) != sample["image_base64_chars"]
                    or row["task"] != task
                    or row["doc_id"] != sample["doc_id"]
                    or row["ignore_eos"] is not False
                ):
                    raise ValueError(f"{path}: sampled question or PNG differs from registration")
                row["at"] = scheduled
                rows.append(row)
                scheduled += arrivals.expovariate(plan["qps"])
        output = run_dir / f"quality-{task}-transport-probe-{plan['qps']:g}qps.trace.jsonl"
        with output.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(output.name, len(rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.selection.read_text(encoding="utf-8"))
    extract(args.run_dir, plan)


if __name__ == "__main__":
    main()
