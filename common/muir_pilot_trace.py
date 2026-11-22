"""Extract frozen MuirBench pilot pairs without repacking the images."""

import argparse
import json
import math
from pathlib import Path
from random import Random


def make_trace(source, plan, qps, output):
    selected = {}
    for first, second in plan["pairs"]:
        if (
            first["idx"] == second["idx"]
            or first["idx"] != second["counterpart_idx"]
            or second["idx"] != first["counterpart_idx"]
            or first["task"] != second["task"]
        ):
            raise ValueError("Invalid pilot pair")
        for row in (first, second):
            doc = row["doc_id"]
            if doc in selected:
                raise ValueError(f"Duplicate selected document {doc}")
            selected[doc] = row
    if not selected or len(selected) != plan["rows"]:
        raise ValueError("Pilot size differs from the plan")

    samples = {}
    with source.open("rb") as stream:
        for doc_id, line in enumerate(stream):
            if doc_id not in selected:
                continue
            row = json.loads(line)
            item = selected[doc_id]
            if (
                row["doc_id"] != doc_id
                or row["task"] != "muirbench"
                or row["dataset_revision"] != plan["revision"]
                or row["idx"] != item["idx"]
                or row["counterpart_idx"] != item["counterpart_idx"]
                or len(row["images"]) != item["images"]
                or len(row["image_digests"]) != item["images"]
            ):
                raise ValueError(f"Frozen trace does not match document {doc_id}")
            samples[doc_id] = row
            if len(samples) == len(selected):
                break
    if samples.keys() != selected.keys():
        raise ValueError("Frozen trace is missing selected documents")

    arrivals = Random(plan["arrival_seed"])
    at = 0.0
    with output.open("x", encoding="utf-8") as stream:
        for doc_id in selected:
            row = samples[doc_id]
            row["at"] = at
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            at += arrivals.expovariate(qps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-trace", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--qps", type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.selection.read_text(encoding="utf-8"))
    qps = plan["candidate_qps"] if args.qps is None else args.qps
    if not math.isfinite(qps) or qps <= 0:
        parser.error("QPS must be finite and positive")
    make_trace(args.frozen_trace, plan, qps, args.output)


if __name__ == "__main__":
    main()
