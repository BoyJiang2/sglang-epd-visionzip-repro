"""Summarize paired B/C70/C50 client, PD, and encoder stage timing from a manifest.

Each configurations entry has a name, optional expected_requests, and groups.
Each group supplies responses, a pd_export path list, and e_metrics_before and
e_metrics_after files. Paths are relative to the manifest directory.
"""

import argparse
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epd_replay import percentiles
from report_performance import fingerprint


GROUPS = ("B", "C70", "C50")
PD_FIELDS = {
    "queue_time": "pd_queue_ms",
    "prefill_latency": "pd_prefill_ms",
    "first_token_latency": "pd_first_token_ms",
    "before_queue": "pd_before_queue_ms",
    "after_prefill": "pd_after_prefill_ms",
}
E_HISTOGRAMS = {
    "queue_wait": ("per_request", 'modality="image"'),
    "preprocess": ("per_encode_batch", 'modality="image"'),
    "model_forward": ("per_encode_batch", 'modality="image"'),
    "transfer": ("per_embedding_send", 'backend="zmq_to_scheduler"'),
    "request_e2e_latency": ("per_encoder_request_before_send", 'modality="image"'),
}
E_SAMPLE = re.compile(
    r"^sglang:encoder_(\w+)_seconds_(count|sum)\{([^}]*)\}\s+(\S+)(?:\s+\S+)?$"
)


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def duration(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def encoder_samples(path):
    samples = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            match = E_SAMPLE.fullmatch(line.rstrip())
            if not match:
                continue
            name, stat, labels, value = match.groups()
            if name not in E_HISTOGRAMS or E_HISTOGRAMS[name][1] not in labels:
                continue
            key = (name, labels)
            stats = samples.setdefault(key, {})
            value = float(value)
            if not duration(value) or stat in stats or (stat == "count" and not value.is_integer()):
                raise ValueError(f"{path}: invalid or duplicate sample for {name} {labels} {stat}")
            stats[stat] = value
    for key, stats in samples.items():
        if set(stats) != {"count", "sum"}:
            raise ValueError(f"{path}: unpaired sum and count for {key}")
    return samples


def encoder_deltas(before_path, after_path):
    before, after = map(encoder_samples, (before_path, after_path))
    if before.keys() - after.keys():
        raise ValueError(f"{after_path}: encoder series disappeared within the replay window")
    deltas = {name: dict(observation_unit=unit, series_count=0, count_delta=0, sum_delta_s=0.0)
              for name, (unit, _) in E_HISTOGRAMS.items()}
    for (name, labels), current in after.items():
        previous = before.get((name, labels), {"count": 0, "sum": 0.0})
        count = int(current["count"] - previous["count"])
        total = current["sum"] - previous["sum"]
        if count < 0 or total < 0 or (count == 0 and total != 0):
            raise ValueError(f"{after_path}: inconsistent cumulative encoder series {name} {labels}")
        result = deltas[name]
        result["series_count"] += 1
        result["count_delta"] += count
        result["sum_delta_s"] += total
    for result in deltas.values():
        count = result["count_delta"]
        result["mean_ms"] = 1000 * result["sum_delta_s"] / count if count else None
        if not result["series_count"]:
            result["count_delta"] = result["sum_delta_s"] = None
    return deltas


def collect_group(base, group, config, expected):
    responses_path = base / config["responses"]
    responses = list(read_jsonl(responses_path))
    signatures = {row["request_id"]: fingerprint(row) for row in responses}
    engine_ids = {row["engine_request_id"] for row in responses}
    if (not responses or len(signatures) != len(responses) or len(engine_ids) != len(responses)
            or len({row["wall_origin_ns"] for row in responses}) != 1
            or (expected is not None and len(responses) != expected)):
        raise ValueError(f"{responses_path}: unexpected request count, duplicate ID or mixed replays")

    exported = {}
    for file in config["pd_export"]:
        for row in read_jsonl(base / file):
            rid = row.get("id")
            if rid not in engine_ids:
                continue
            if rid in exported:
                raise ValueError(f"{file}: duplicate PD timing record for {rid}")
            exported[rid] = row

    requests = []
    for response in responses:
        rid = response["engine_request_id"]
        status = response["status"]
        images = response["image_digests"]
        planned = response["planned_offset_s"]
        if (response["group"] != group or status not in ("complete", "failed")
                or not rid or not images or len(images) != response["image_count"] or not duration(planned)):
            raise ValueError(f"{responses_path}: invalid replay row for {rid}")
        chunks = response["content_chunks"]
        client_ttft = chunks[0][0] if chunks else None
        if status == "complete" and not duration(client_ttft):
            raise ValueError(f"{responses_path}: missing first content time for {rid}")
        pd = exported.get(rid, {})
        metrics = {dest: (1000 * pd[src] if duration(pd.get(src)) else None)
                   for src, dest in PD_FIELDS.items()}
        metrics["client_ttft_ms"] = 1000 * client_ttft if duration(client_ttft) else None
        queue, prefill, first = (pd.get(field) for field in ("queue_time", "prefill_latency", "first_token_latency"))
        remainder = first - queue - prefill if all(map(duration, (first, queue, prefill))) else None
        metrics["pd_first_token_other_ms"] = (
            1000 * remainder if pd.get("num_retractions") == 0 and duration(remainder) else None
        )
        requests.append(dict(
            request_id=response["request_id"], engine_request_id=rid, status=status,
            planned_offset_s=planned, image_digests=images, pd_matched=rid in exported,
            num_retractions=pd.get("num_retractions"), **metrics,
        ))

    complete = [row for row in requests if row["status"] == "complete"]
    result = dict(
        replay_rows=len(requests), complete=len(complete), failed=len(requests) - len(complete),
        pd_matched=len(exported), complete_pd_matched=sum(row["pd_matched"] for row in complete),
        complete_latencies_ms={name: percentiles([row[name] for row in complete if row[name] is not None])
                               for name in metrics},
        encoder_histograms=encoder_deltas(base / config["e_metrics_before"], base / config["e_metrics_after"]),
        requests=requests,
    )
    return signatures, result


def report(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configurations = manifest["configurations"]
    if not configurations or len({entry["name"] for entry in configurations}) != len(configurations):
        raise ValueError("Expected nonempty, uniquely named configurations")
    results = []
    for entry in configurations:
        name, groups = entry["name"], entry["groups"]
        if set(groups) != set(GROUPS):
            raise ValueError(f"{name}: expected groups {GROUPS}")
        reference = None
        rendered = {}
        for group in GROUPS:
            signature, rendered[group] = collect_group(
                manifest_path.parent, group, groups[group], entry.get("expected_requests")
            )
            if reference is None:
                reference = signature
            elif signature != reference:
                raise ValueError(f"{name}/{group}: request IDs, images, output lengths or planned arrivals differ")
        results.append(dict(name=name, groups=rendered))
    return {"configurations": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.manifest)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
