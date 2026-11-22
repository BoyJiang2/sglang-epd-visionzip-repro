"""Verify the frozen synthetic matrix and plot the common-load comparison."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from epd_replay import percentiles, summarize_run


def load_cell(path, group, expected):
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rid = row["request_id"]
            if row["group"] != group or rid in rows:
                raise ValueError(f"{path}: incorrect group or duplicate request {rid}")
            rows[rid] = row
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} request records, got {len(rows)}")
    return rows


def fingerprint(row):
    return tuple(row[key] for key in (
        "planned_offset_s", "image_count", "image_digests", "expected_output_tokens",
    ))


def collect(run_dir, plan, b_label):
    groups = plan["groups"]
    rates = {int(images): row for images, row in plan["rates_qps"].items()}
    common_rates = set.intersection(*(set(row) for row in rates.values()))
    if len(common_rates) != 1:
        raise ValueError("The main figure needs one frozen QPS common to all image counts")
    common_rate, = common_rates
    paper = plan["paper"]
    cases = [
        ("formal", images, rate, seed, plan["requests_per_cell"], plan["output_tokens"])
        for images, options in rates.items()
        for rate in options
        for seed in plan["seeds"]
    ]
    cases.extend(
        ("official", paper["random_images_upper_bound"], rate, paper["seed"], paper["requests"], paper["output_tokens"])
        for rate in paper["target_rates"]
    )
    runs = []
    pooled = defaultdict(list)
    for kind, images, rate, seed, count, output_tokens in cases:
        name = f"{kind}-{images}img-{rate}qps-seed{seed}"
        first = None
        for group in groups:
            suffix = f".{b_label}" if group == "B" and b_label else ""
            path = run_dir / f"{name}.{group}{suffix}.jsonl"
            rows = load_cell(path, group, count)
            signatures = {rid: fingerprint(row) for rid, row in rows.items()}
            if any(
                row["expected_output_tokens"] != output_tokens
                or not 1 <= row["image_count"] <= images
                or row["image_count"] != len(row["image_digests"])
                or (kind == "formal" and row["image_count"] != images)
                for row in rows.values()
            ):
                raise ValueError(f"{path}: workload differs from the frozen plan")
            if first is None:
                first = signatures
            elif first != signatures:
                raise ValueError(f"{path}: inputs or planned arrivals differ across groups")
            metrics = summarize_run(path, count)
            runs.append(dict(kind=kind, images=images, qps=rate, seed=seed, **metrics))
            if kind == "formal" and rate == common_rate:
                pooled[group, images].extend(rows.values())

    figure = []
    for (group, images), rows in sorted(pooled.items()):
        ttft = [row["content_chunks"][0][0] * 1000 for row in rows if row["status"] == "complete"]
        figure.append(dict(
            group=group, images=images, submitted=len(rows), completed=len(ttft),
            completion_rate=len(ttft) / len(rows), ttft_ms=percentiles(ttft),
        ))
    return {"common_qps": common_rate, "groups": groups, "runs": runs, "figure": figure}


def plot(report, destination):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    fig = Figure(figsize=(8.4, 3.4), layout="constrained")
    FigureCanvasAgg(fig)
    latency, completion = fig.subplots(1, 2)
    styles = {"A": ("#585858", "o"), "B": ("#26618B", "s"), "C70": ("#AE502B", "^"), "C50": ("#30806D", "D")}
    for group in report["groups"]:
        rows = [row for row in report["figure"] if row["group"] == group]
        x = [row["images"] for row in rows]
        color, marker = styles[group]
        p95 = [row["ttft_ms"]["p95"] for row in rows]
        latency.plot(x, [v / 1000 if v is not None else float("nan") for v in p95],
                     label=group, color=color, marker=marker)
        completion.plot(x, [row["completion_rate"] for row in rows], color=color, marker=marker)
    latency.set(yscale="log", ylabel="P95 TTFT (s, log scale)")
    completion.set(ylim=(0, 1.05), ylabel="Completed requests")
    completion.yaxis.set_major_formatter(PercentFormatter(1))
    for axes in (latency, completion):
        axes.set(xlabel="Images per request", xticks=sorted({row["images"] for row in report["figure"]}))
        axes.grid(axis="y", alpha=0.2)
    latency.legend(title="Configuration")
    fig.suptitle(f"Qwen2.5-VL-7B · {report['common_qps']:g} requests/s")
    with destination.open("xb") as stream:
        fig.savefig(stream, format=destination.suffix.lstrip("."), dpi=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--b-label", default="no-probe")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    if args.b_label and not args.b_label.replace("-", "").isalnum():
        parser.error("B label must contain letters, numbers or hyphens")
    if args.figure.suffix not in (".pdf", ".png") or args.figure == args.output:
        parser.error("the figure needs a separate .pdf or .png path")
    if args.output.exists() or args.figure.exists():
        parser.error("choose new paths for the report and figure")
    plan = json.loads((args.run_dir / "formal-design.json").read_text(encoding="utf-8"))
    report = collect(args.run_dir, plan, args.b_label)
    plot(report, args.figure)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
