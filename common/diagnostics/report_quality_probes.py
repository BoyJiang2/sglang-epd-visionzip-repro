"""Plot measured E/PD traffic on paired image probes against full-set quality."""

import argparse
import base64
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
from random import Random
import runpy
from statistics import mean

from report_muir_traffic import NATIVE_TRACE, PD_TRACE, VISIONZIP_TRACE, index_log, traffic


TASK_NAMES = {"ocrbench": "OCRBench", "realworldqa": "RealWorldQA"}
GROUPS = ("B", "C70", "C50")
SAMPLES = 32
DATASETS = runpy.run_path(str(Path(__file__).resolve().parents[1] / "epd_replay.py"))["QUALITY_DATASETS"]


def read_registered(run_dir, plan, task):
    info = plan["tasks"][task]
    source = run_dir / info["source"]
    stat = source.stat()
    if (stat.st_size, stat.st_mtime_ns) != (info["source_size_bytes"], info["source_mtime_ns"]):
        raise ValueError(f"{source}: frozen source differs from registration")
    _, revision, dataset_size = DATASETS[task]
    samples = info["samples"]
    if len(samples) != SAMPLES or len({sample["file_offset_bytes"] for sample in samples}) != SAMPLES:
        raise ValueError(f"{task}: expected {SAMPLES} distinct registered offsets")
    rng = Random(plan["arrival_seed"])
    scheduled = 0.0
    registered = {}
    with source.open("rb") as stream:
        for sample in samples:
            doc_id = sample["doc_id"]
            rid = f"{task}-{doc_id:05d}"
            if rid in registered or not 0 <= doc_id < dataset_size:
                raise ValueError(f"{source}: invalid or duplicate registered question {doc_id}")
            stream.seek(sample["file_offset_bytes"])
            row = json.loads(stream.readline())
            image, = row["images"]
            prefix, _, encoded = image.partition(",")
            digest = hashlib.sha256(base64.b64decode(encoded, validate=True)).hexdigest()
            if (
                (row["id"], row["task"], row["doc_id"], row["dataset_revision"])
                != (rid, task, doc_id, revision)
                or prefix != "data:image/png;base64"
                or len(encoded) != sample["image_base64_chars"]
                or row["image_digests"] != [digest]
                or row["ignore_eos"] is not False
            ):
                raise ValueError(f"{source}: registered image or question differs at offset {sample['file_offset_bytes']}")
            registered[rid] = dict(sample, image_digests=[digest], planned_offset_s=scheduled)
            scheduled += rng.expovariate(plan["qps"])
    return registered


def read_responses(path, task, group, registered):
    _, revision, _ = DATASETS[task]
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rid = row["request_id"]
            expected = registered.get(rid)
            if (
                rid in rows
                or not expected
                or (row["group"], row["task"], row["dataset_revision"]) != (group, task, revision)
                or row["doc_id"] != expected["doc_id"]
                or row["image_count"] != 1
                or row["image_digests"] != expected["image_digests"]
                or not math.isclose(row["planned_offset_s"], expected["planned_offset_s"], abs_tol=1e-12)
            ):
                raise ValueError(f"{path}: response {rid} does not match the registered probe")
            rows[rid] = row
    if set(rows) != set(registered):
        raise ValueError(f"{path}: expected all {SAMPLES} registered response records")
    return rows


def full_score(path, task, group):
    _, revision, dataset_size = DATASETS[task]
    scores = json.loads(path.read_text(encoding="utf-8"))
    items = scores["items"]
    completed = sum(item["status"] == "complete" for item in items)
    correct = sum(item["correct"] is True for item in items)
    official = scores["official_score"]
    submitted_accuracy = correct / dataset_size
    if (
        (scores["task"], scores["dataset_revision"], scores["group"]) != (task, revision, group)
        or scores["completed"] != completed
        or scores["submitted"] != len(items)
        or len(items) != dataset_size
        or {item["doc_id"] for item in items} != set(range(dataset_size))
        or any(
            item["status"] not in ("complete", "failed", "skipped")
            or (type(item["correct"]) is not bool if item["status"] == "complete" else item["correct"] is not None)
            or item["request_id"] != f"{task}-{item['doc_id']:05d}"
            for item in items
        )
        or scores["correct"] != correct
        or not math.isclose(scores["service_accuracy"], submitted_accuracy)
        or (completed == dataset_size and (official is None or not math.isclose(official, submitted_accuracy)))
        or (completed != dataset_size and official is not None)
    ):
        raise ValueError(f"{path}: scores must account for every question in the pinned test set")
    return dict(submitted=dataset_size, completed=completed, correct=correct,
                submitted_accuracy=submitted_accuracy, full_set_accuracy=official)


def report(run_dir, plan, paths, client_hosts):
    if plan["qps"] != 2 or set(plan["tasks"]) != set(TASK_NAMES):
        raise ValueError("Expected the registered two-task, 2 QPS quality probe")
    registered = {task: read_registered(run_dir, plan, task) for task in TASK_NAMES}
    result = {
        "qps": plan["qps"],
        "arrival_seed": plan["arrival_seed"],
        "registered_requests_per_task": SAMPLES,
        "bytes_scope": "PD-to-E HTTP request body plus E-to-PD embedding and metadata frames.",
        "tasks": {task: {"groups": {}} for task in TASK_NAMES},
    }
    for group, (encoder_log, pd_log) in paths.items():
        responses = {
            task: read_responses(
                run_dir / f"quality-{task}-transport-probe-2qps.{group}.jsonl",
                task, group, registered[task],
            )
            for task in TASK_NAMES
        }
        engine_ids = [row["engine_request_id"] for rows in responses.values() for row in rows.values()]
        if len(engine_ids) != len(set(engine_ids)):
            raise ValueError(f"{group}: duplicate engine request IDs")
        encoder = index_log(encoder_log, NATIVE_TRACE if group == "B" else VISIONZIP_TRACE, set(engine_ids))
        pd = index_log(pd_log, PD_TRACE, set(engine_ids))
        for task in TASK_NAMES:
            score = full_score(run_dir / f"quality-{task}.{group}.scores.json", task, group)
            samples = {
                rid: dict(
                    expected,
                    status=responses[task][rid]["status"],
                    engine_request_id=responses[task][rid]["engine_request_id"],
                    **traffic(group, responses[task][rid], client_hosts, encoder, pd),
                )
                for rid, expected in registered[task].items()
            }
            result["tasks"][task]["groups"][group] = dict(score, requests=samples)
    total_key = "cross_node_total_application_bytes"
    for task, result_task in result["tasks"].items():
        _, revision, dataset_size = DATASETS[task]
        measurements = result_task["groups"]
        paired = [
            rid for rid in registered[task]
            if all(group["requests"][rid][total_key] is not None for group in measurements.values())
        ]
        result_task.update(
            dataset_revision=revision,
            full_set_rows=dataset_size,
            paired_measured_requests=len(paired),
            paired_measured_ids=paired,
        )
        for group in measurements.values():
            group["measured_requests"] = sum(row[total_key] is not None for row in group["requests"].values())
            group["mean_measured_mib_per_request"] = (
                mean(group["requests"][rid][total_key] for rid in paired) / (1024 ** 2) if paired else None
            )
    return result


def make_figure(result):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), layout="constrained")
    colors = ("#4A5962", "#007F75", "#315DAB")
    for axis, (task, name) in zip(axes, TASK_NAMES.items()):
        data = result["tasks"][task]
        points = data["groups"]
        paired = data["paired_measured_requests"]
        axis.set_title(f"{name} · {data['full_set_rows']:,} test questions", loc="left", weight="bold")
        axis.set_xlabel(f"Mean measured application MiB/request\n{paired}/{SAMPLES} paired probes at 2 QPS")
        axis.set_ylabel("Correct / submitted (%)")
        if paired:
            x = [points[group]["mean_measured_mib_per_request"] for group in GROUPS]
            y = [100 * points[group]["submitted_accuracy"] for group in GROUPS]
            axis.plot(x, y, color="#9BA7AD", linewidth=1, zorder=1)
            for group, xp, yp, color in zip(GROUPS, x, y, colors):
                score = points[group]
                failed = score["submitted"] - score["completed"]
                label = f"{group} ({failed} failed or skipped)" if failed else group
                axis.scatter(xp, yp, c=color, label=label, s=52, zorder=2)
            axis.legend(frameon=False, title="Configuration")
            axis.grid(alpha=0.22)
            axis.margins(x=0.13, y=0.22)
        else:
            axis.text(0.5, 0.5, "No probes with measurements in all groups", ha="center", transform=axis.transAxes)
    image = BytesIO()
    fig.savefig(image, format="png", dpi=200)
    plt.close(fig)
    return image.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--selection", type=Path, help="Defaults to quality-transport-probe-selection.json in the run directory.")
    parser.add_argument("--client-host", required=True, action="append", help="Local PD host or IP; repeat for aliases.")
    for group in GROUPS:
        parser.add_argument(f"--{group.lower()}-logs", required=True, type=Path, nargs=2, metavar=("ENCODER_LOG", "PD_LOG"))
    parser.add_argument("--output", required=True, type=Path, help="New JSON report.")
    parser.add_argument("--figure", required=True, type=Path, help="New two-panel PNG figure.")
    args = parser.parse_args()
    if args.figure.suffix.lower() != ".png":
        parser.error("--figure must have a .png extension")
    if args.output.resolve() == args.figure.resolve() or args.output.exists() or args.figure.exists():
        parser.error("provide two distinct, unused output paths")
    selection = args.selection or args.run_dir / "quality-transport-probe-selection.json"
    plan = json.loads(selection.read_text(encoding="utf-8"))
    paths = {group: getattr(args, f"{group.lower()}_logs") for group in GROUPS}
    client_hosts = {host.casefold().rstrip(".") for host in args.client_host}
    joined = report(args.run_dir, plan, paths, client_hosts)
    image = make_figure(joined)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(joined, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    with args.figure.open("xb") as stream:
        stream.write(image)


if __name__ == "__main__":
    main()
