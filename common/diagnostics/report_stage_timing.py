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
RPC_TRACE = re.compile(
    r"\bepd_encode_http req_id=(\S+) part_idx=(\d+) .*?http_status=(\S+)(?: rpc_headers_ms=(\S+))?\s*$"
)
NATIVE_SERIAL_TRACE = re.compile(
    r"\bepd_feature_payload req_id=(\S+) rows=\d+ feature_bytes=\d+ metadata_bytes=\d+ serialize_ms=(\S+)\s*$"
)
BATCH_PHASES = {
    "vit_ms": "CUDA ViT forward through native merger and reorder, excluding score",
    "score_ms": "CUDA attention-statistics interval inside the final ViT block",
    "merge_ms": "CUDA selection, token merge, and concatenation after ViT",
    "feature_to_host_wall_ms": "host wall time enclosing feature computation and blocking D2H copy",
}


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def duration(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def selected_part(req_id, part_idx, engine_ids, path):
    engine_id, _, suffix = req_id.rpartition("_local_part_")
    if engine_id not in engine_ids:
        if req_id in engine_ids:
            raise ValueError(f"{path}: missing encoder part suffix for {req_id}")
        return None
    if not suffix.isdecimal() or (part_idx is not None and int(suffix) != part_idx):
        raise ValueError(f"{path}: inconsistent encoder part for {req_id}")
    return engine_id, int(suffix)


def put_unique(target, key, value, path):
    if key in target:
        raise ValueError(f"{path}: duplicate phase record for {key}")
    target[key] = value


def read_rpc_log(path, engine_ids):
    records = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            match = RPC_TRACE.search(line)
            if not match:
                continue
            req_id, part_idx, status, elapsed = match.groups()
            part = selected_part(req_id, int(part_idx), engine_ids, path)
            if part is None:
                continue
            elapsed = float(elapsed) if elapsed is not None else None
            if (status != "exception" and (not status.isdecimal() or not 100 <= int(status) <= 599)
                    or elapsed is not None and not duration(elapsed)):
                raise ValueError(f"{path}: invalid RPC outcome or time for {req_id}")
            put_unique(records, part, dict(http_status=status, rpc_headers_ms=elapsed), path)
    return records


def read_encoder_log(path, group, engine_ids):
    batches, serial, batch_parts = {}, {}, {}
    ratio = int(group[1:]) if group != "B" else 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if "visionzip_phase_batch " in line:
                row = json.loads(line.split("visionzip_phase_batch ", 1)[1])
                selected = {}
                for req_id, part_idx, num_parts in row["req_parts"]:
                    part = selected_part(req_id, part_idx, engine_ids, path)
                    if part is not None:
                        if type(num_parts) is not int or not 0 <= part[1] < num_parts:
                            raise ValueError(f"{path}: invalid part count for {req_id}")
                        put_unique(selected, part, num_parts, path)
                if selected:
                    if row["ratio"] != ratio or not all(duration(row.get(field)) for field in BATCH_PHASES):
                        raise ValueError(f"{path}: invalid batch ratio or duration for {row['batch_id']}")
                    put_unique(batches, row["batch_id"], (row, selected), path)
                    for part, num_parts in selected.items():
                        put_unique(batch_parts, part, num_parts, path)
                continue
            if "visionzip_phase_serialize " in line:
                row = json.loads(line.split("visionzip_phase_serialize ", 1)[1])
                part = selected_part(row["req_id"], row["part_idx"], engine_ids, path)
                if part is None:
                    continue
                if (row["ratio"] != ratio or not duration(row["serialize_ms"])
                        or type(row["num_parts"]) is not int or not 0 <= part[1] < row["num_parts"]):
                    raise ValueError(f"{path}: invalid VisionZip serialization phase for {part}")
                put_unique(serial, part, row, path)
            elif group == "B" and (match := NATIVE_SERIAL_TRACE.search(line)):
                req_id, elapsed = match.groups()
                part = selected_part(req_id, None, engine_ids, path)
                if part is not None:
                    elapsed = float(elapsed)
                    if not duration(elapsed):
                        raise ValueError(f"{path}: invalid native serialization phase for {part}")
                    put_unique(serial, part, dict(req_id=req_id, part_idx=part[1], serialize_ms=elapsed), path)
    for part in batch_parts.keys() & serial.keys():
        if serial[part]["num_parts"] != batch_parts[part]:
            raise ValueError(f"{path}: conflicting batch and serialization part counts for {part}")
    return batches, serial


def attach_phase_diagnostics(results, pd_logs, encoder_logs):
    rpc_index, encoder_index = {}, {}
    for group in pd_logs.keys() | encoder_logs.keys():
        ids = {row["engine_request_id"] for case in results for row in case["groups"][group]["requests"]}
        if group in pd_logs:
            rpc_index[group] = read_rpc_log(pd_logs[group], ids)
        if group in encoder_logs:
            encoder_index[group] = read_encoder_log(encoder_logs[group], group, ids)

    for case in results:
        groups = case["groups"]
        complete = set.intersection(*(
            {row["request_id"] for row in item["requests"] if row["status"] == "complete"}
            for item in groups.values()
        ))
        ids = {group: {row["engine_request_id"]: row["request_id"] for row in item["requests"]}
               for group, item in groups.items()}
        rpc = {group: {(ids[group][rid], part): row for (rid, part), row in records.items() if rid in ids[group]}
               for group, records in rpc_index.items()}
        common_rpc = set.intersection(*(
            {key for key, row in records.items() if key[0] in complete and row["http_status"] == "200"
             and row["rpc_headers_ms"] is not None} for records in rpc.values()
        )) if set(rpc) == set(GROUPS) else None
        for group in pd_logs.keys() | encoder_logs.keys():
            detail = dict(common_completed_request_count=len(complete))
            groups[group]["phase_diagnostics"] = detail
            if group in rpc:
                parts = [dict(request_id=rid, part_idx=part, **row) for (rid, part), row in rpc[group].items()]
                successful = [row for row in parts if row["http_status"] == "200"]
                failed = [row for row in parts if row["http_status"] != "200"]
                detail["pd_rpc"] = dict(
                    observation_unit="per_encoder_http_part",
                    success_measure="PD HTTP call start to response headers",
                    failure_measure="PD HTTP call start to non-200 headers or exception",
                    matched_request_count=len({row["request_id"] for row in parts}),
                    matched_part_count=len(parts), successful_part_count=len(successful), failed_part_count=len(failed),
                    successful_rpc_headers_ms=percentiles([row["rpc_headers_ms"] for row in successful
                                                          if row["rpc_headers_ms"] is not None]),
                    failed_rpc_elapsed_ms=percentiles([row["rpc_headers_ms"] for row in failed
                                                      if row["rpc_headers_ms"] is not None]),
                    common_successful_rpc_headers_ms=(percentiles([rpc[group][key]["rpc_headers_ms"]
                                                                   for key in common_rpc])
                                                      if common_rpc is not None else None),
                    parts=parts,
                )
            if group in encoder_index:
                batches, serial = encoder_index[group]
                eligible = {rid for rid, logical in ids[group].items() if logical in complete}
                observed = [(row, members) for row, members in batches.values()
                            if any(rid in eligible for rid, _ in members)]
                included = [(row, members) for row, members in observed if len(members) == len(row["req_parts"])
                            and all(rid in eligible for rid, _ in members)]
                serialized = [row for (rid, _), row in serial.items() if rid in eligible]
                detail["encoder"] = dict(
                    batch_observation_unit="per_gpu_encode_batch",
                    batch_measures=BATCH_PHASES,
                    batch_count=len(included), batches_with_common_members=len(observed),
                    mixed_batches_excluded=len(observed) - len(included),
                    batch_request_count=len({rid for _, members in included for rid, _ in members}),
                    batch_part_count=sum(len(members) for _, members in included),
                    **{field: percentiles([row[field] for row, _ in included]) for field in BATCH_PHASES},
                    serialization_observation_unit="per_embedding_part",
                    serialization_measure="E host wall time preparing the embedding frames before send",
                    serialization_request_count=len({rid for rid, _ in serial if rid in eligible}),
                    serialization_part_count=len(serialized),
                    serialize_ms=percentiles([row["serialize_ms"] for row in serialized]),
                    batches=[row for row, _ in included], serialization_parts=serialized,
                )


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


def report(manifest_path, pd_logs=None, encoder_logs=None):
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
    if pd_logs or encoder_logs:
        attach_phase_diagnostics(results, pd_logs or {}, encoder_logs or {})
    return {"configurations": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pd-log", nargs=2, action="append", default=[], metavar=("GROUP", "PATH"))
    parser.add_argument("--encoder-log", nargs=2, action="append", default=[], metavar=("GROUP", "PATH"))
    args = parser.parse_args()
    paths = []
    for entries in (args.pd_log, args.encoder_log):
        mapping = {group: Path(path) for group, path in entries}
        if len(mapping) != len(entries) or mapping.keys() - set(GROUPS):
            parser.error(f"use each group in {GROUPS} at most once per log flag")
        paths.append(mapping)
    result = report(args.manifest, *paths)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
