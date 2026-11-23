"""Report per-request TPOT from eight paired single-image replays."""

import argparse
import json
from itertools import chain
from math import isclose
from pathlib import Path
from random import Random
from statistics import quantiles

from diagnostics.sample_7b_roles import summarize_window
from epd_replay import percentiles, summarize_run
from report_performance import fingerprint, load_cell


def summarize(run, plan):
    groups = plan["groups"]
    formal, extra = plan["formal_seeds"], plan["supplement_seeds"]
    count, tokens = plan["requests_per_replay"], plan["output_tokens"]
    minimum = plan["minimum_completed_per_group"]
    if (groups != ["A", "B", "C70", "C50"] or len(formal) != 3 or len(extra) != 5
            or len(set(formal + extra)) != 8 or (plan["images"], plan["qps"], count, tokens, minimum) != (1, 4, 128, 64, 1000)):
        raise ValueError("Expected the registered 1-image, 4-QPS supplement and 1024 attempts per group")
    samples = {group: [] for group in groups}
    repeats = {group: [] for group in groups}

    for seed in (*formal, *extra):
        core = seed in formal
        prefix = "formal" if core else "tail-p99"
        profile = f"{plan['images']}img-{plan['qps']:g}qps-seed{seed}"
        arrivals = Random(seed)
        scheduled = {}
        at = 0.0
        for index in range(count):
            scheduled[f"{seed}-{index:05d}"] = at
            at += arrivals.expovariate(plan["qps"])
        reference = None
        for group in groups:
            label = "B.no-probe" if core and group == "B" else group
            path = run / f"{prefix}-{profile}.{label}.jsonl"
            rows = load_cell(path, group, count)
            if rows.keys() != scheduled.keys() or any(
                row["image_count"] != plan["images"] or len(row["image_digests"]) != plan["images"]
                or row["expected_output_tokens"] != tokens
                or not isclose(row["planned_offset_s"], scheduled[rid], abs_tol=1e-12, rel_tol=0)
                for rid, row in rows.items()
            ):
                raise ValueError(f"{path}: workload differs from the registered load")
            matched = {rid: fingerprint(row) for rid, row in rows.items()}
            if reference is None:
                reference = matched
            elif reference != matched:
                raise ValueError(f"{path}: inputs or planned arrivals are unpaired")
            metrics = summarize_run(path, count)
            values = [
                (row["duration_s"] - row["content_chunks"][0][0]) * 1000 / (tokens - 1)
                for row in rows.values() if row["status"] == "complete" and row["completion_tokens"] == tokens
            ]
            samples[group].append(values)
            activity = {node: summarize_window(run, f"{group}-{node}-tail-p99", list(rows.values()))
                        for node in ("qjy000", "qjy001")} if not core else None
            repeats[group].append(dict(seed=seed, records=count, completed=metrics["completed"],
                                       failures=metrics["failures_by_type"], tpot_samples=len(values),
                                       sse_gaps=metrics["sse_content_chunk_gap_ms"]["n"], gpu_activity=activity))

    results = {}
    for group, runs in repeats.items():
        timing = percentiles(chain.from_iterable(samples[group]))
        qualified = timing["n"] >= minimum
        ci = None
        if qualified:
            rng = Random(23011)
            estimates = [
                percentiles(chain.from_iterable(rng.choices(cell, k=len(cell)) for cell in samples[group]))["p99"]
                for _ in range(2000)
            ]
            cuts = quantiles(estimates, n=40, method="inclusive")
            ci = [cuts[0], cuts[-1]]
        results[group] = dict(
            records=sum(run["records"] for run in runs), completed=sum(run["completed"] for run in runs),
            qualified=qualified, request_tpot_ms=timing, p99_ci95_ms=ci,
            sse_content_gaps=sum(run["sse_gaps"] for run in runs), repeats=runs,
        )
    return dict(plan=plan, method="2000 bootstrap draws over requests within each seed", groups=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.run_dir, json.loads(args.plan.read_text(encoding="utf-8")))
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return 0 if all(group["qualified"] for group in report["groups"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
