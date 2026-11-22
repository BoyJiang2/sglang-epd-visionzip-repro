"""Report long-request SSE content-block gaps around a heavy image arrival."""

import argparse
from collections import Counter
import json
from pathlib import Path

from epd_replay import percentiles


def analyze_mode(run_dir, group, mode, repeats):
    long_statuses, heavy_statuses = Counter(), Counter()
    pre, post = [], []
    per_repeat = []
    for i in range(1, repeats + 1):
        path = run_dir / f"itl-{group}-{mode}-{i}.jsonl"
        with path.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        by_role = {row["interference_role"]: row for row in rows}
        roles = {"long"} if mode == "control" else {"long", "heavy"}
        if len(rows) != len(roles) or by_role.keys() != roles or any(row["group"] != group for row in rows):
            raise ValueError(f"{path}: unexpected request identities or roles")
        long, heavy = by_role["long"], by_role.get("heavy")
        if long["expected_output_tokens"] != 256 or long["image_count"] != 1:
            raise ValueError(f"{path}: expected a one-image, 256-token long request")
        long_statuses[long["status"]] += 1
        if heavy is not None:
            if heavy["image_count"] != 8 or (heavy["endpoint"] == long["endpoint"]) != (mode == "same"):
                raise ValueError(f"{path}: heavy request differs from the planned routing")
            heavy_statuses[heavy["status"]] += 1

        row = dict(run=i, long_status=long["status"], long_output_tokens=long["completion_tokens"],
                   heavy_status=heavy["status"] if heavy is not None else None)
        per_repeat.append(row)
        if long["status"] != "complete" or long["completion_tokens"] != 256:
            continue
        chunks = [long["send_offset_s"] + chunk[0] for chunk in long["content_chunks"]]
        if not chunks or any(right < left for left, right in zip(chunks, chunks[1:])):
            raise ValueError(f"{path}: invalid long-stream timestamps")
        trigger = chunks[0] + 0.05 if heavy is None else heavy["send_offset_s"]
        if trigger is None:
            continue
        if trigger < chunks[0]:
            raise ValueError(f"{path}: heavy request was sent before the long stream started")
        row["aligned_t_s"] = trigger
        after = []
        for left, right in zip(chunks, chunks[1:]):
            gap = (right - left) * 1000
            # Include the interval that straddles the arrival; it can contain a stall.
            (after if right >= trigger else pre).append(gap)
        row["post_trigger_gap_ms"] = percentiles(after)
        post.extend(after)
    return {
        "submitted_repeats": repeats,
        "long_outcomes": dict(long_statuses),
        "heavy_outcomes": dict(heavy_statuses),
        "before_gap_ms": percentiles(pre),
        "after_gap_ms": percentiles(post),
        "per_repeat": per_repeat,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", choices=("A", "B", "C70", "C50"), required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error("repeats must be positive")
    report = {
        "measure": "Adjacent nonempty SSE content-block gaps within the same long request",
        "alignment": "Actual heavy request send; nominal first content + 50 ms in control",
        "inclusion": "Completed 256-token long requests with usable alignment; pooled within each mode",
        "groups": {
            group: {
                mode: analyze_mode(args.run_dir, group, mode, args.repeats)
                for mode in (("control", "same", "separate") if group == "A" else ("control", "same"))
            }
            for group in args.groups
        },
    }
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
