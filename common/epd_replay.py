"""Freeze and replay a SGLang image workload through one or more endpoints."""

import argparse
import asyncio
import base64
import hashlib
import json
import math
import random
import time
import uuid
from collections import Counter, defaultdict
from contextlib import ExitStack
from io import BytesIO
from itertools import chain
from pathlib import Path
from types import SimpleNamespace


QUALITY_DATASETS = {
    "ocrbench": ("echo840--OCRBench", "92a54bd1384387c178d5a07140a2d85e0a3d12e1", 1000),
    "realworldqa": ("lmms-lab--RealWorldQA", "907c4e5228fd1703c710ed937601cb5f89ab8d5c", 765),
    "muirbench": ("MUIRBENCH--MUIRBENCH", "4c393cffc985c77d28de3b9045e2e5186920df80", 2600),
}


def generate(args):
    import numpy as np
    from sglang.benchmark.datasets.image import sample_image_requests
    from sglang.benchmark.utils import get_processor

    processor = get_processor(args.model)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rows = sample_image_requests(
        num_requests=args.requests,
        image_count=args.images,
        input_len=128,
        output_len=args.output_tokens,
        range_ratio=1.0,
        processor=processor,
        image_content="random",
        image_format="jpeg",
        image_resolution="1080p",
        backend="sglang-oai-chat",
        random_image_count=args.random_images,
    )
    if len(rows) != args.requests or any(row.output_len != args.output_tokens for row in rows):
        raise ValueError("SGLang generated an unexpected request count or output length")
    arrivals = random.Random(args.seed)
    at = 0.0
    with args.trace.open("x", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            images = row.image_data
            assert isinstance(images, list) and all(
                image.startswith("data:image/jpeg;base64,") for image in images
            ), "unexpected SGLang image representation"
            request = {
                "id": f"{args.seed}-{index:05d}",
                "at": at,
                "text": row.prompt,
                "images": images,
                "image_digests": [
                    hashlib.sha256(base64.b64decode(image.split(",", 1)[1])).hexdigest()
                    for image in images
                ],
                "vision_tokens_estimate": row.vision_prompt_len,
                "output_tokens": row.output_len,
            }
            stream.write(json.dumps(request, ensure_ascii=False) + "\n")
            at += arrivals.expovariate(args.qps)


def load_quality_task(args):
    import lmms_eval.api.metrics  # Register the fixed evaluator's exact-match metric.
    from lmms_eval.api.task import ConfigurableTask
    from lmms_eval.utils import load_yaml_config

    repo, revision, expected_rows = QUALITY_DATASETS[args.task]
    snapshot = args.hub_cache / f"datasets--{repo}" / "snapshots" / revision
    shards = [str(path) for path in sorted((snapshot / "data").glob("*.parquet"))]
    if not shards:
        raise FileNotFoundError(f"No fixed dataset shards in {snapshot}")
    yaml = args.lmms_eval / "lmms_eval" / "tasks" / args.task / f"{args.task}.yaml"
    config = load_yaml_config(yaml_path=str(yaml))
    config["dataset_path"] = "parquet"
    config["dataset_kwargs"] = {"data_files": {"test": shards}}
    task = ConfigurableTask(config=config, model_name="openai_compatible")
    if len(task.eval_docs) != expected_rows:
        raise ValueError(f"{args.task}: expected {expected_rows} rows, got {len(task.eval_docs)}")
    return task, revision


def quality_images(task, doc):
    urls, digests = [], []
    for image in task.doc_to_visual(doc):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        urls.append("data:image/png;base64," + base64.b64encode(data).decode("ascii"))
        digests.append(hashlib.sha256(data).hexdigest())
    return urls, digests


def generate_quality(args):
    task, revision = load_quality_task(args)
    docs = task.eval_docs
    count = args.limit if args.limit is not None else len(docs)
    if count > len(docs):
        raise ValueError(f"Limit {count} exceeds {len(docs)} rows")
    arrivals = random.Random(args.seed)
    at = 0.0
    with args.trace.open("x", encoding="utf-8") as stream:
        for doc_id in range(count):
            doc = docs[doc_id]
            urls, digests = quality_images(task, doc)
            request = {
                "id": f"{args.task}-{doc_id:05d}",
                "at": at,
                "text": task.doc_to_text(doc),
                "images": urls,
                "image_digests": digests,
                "vision_tokens_estimate": None,
                "output_tokens": task.config.generation_kwargs["max_new_tokens"],
                "ignore_eos": False,
                "task": args.task,
                "dataset_revision": revision,
                "doc_id": doc_id,
            }
            if args.task == "muirbench":
                request.update(idx=doc["idx"], counterpart_idx=doc["counterpart_idx"])
            stream.write(json.dumps(request, ensure_ascii=False) + "\n")
            at += arrivals.expovariate(args.qps)


def score_quality(args):
    task, revision = load_quality_task(args)
    docs = task.eval_docs_no_media
    source = {}
    with args.trace.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            doc_id = row["doc_id"]
            if (
                row["task"] != args.task
                or row["dataset_revision"] != revision
                or row["id"] != f"{args.task}-{doc_id:05d}"
                or row["id"] in source
                or not 0 <= doc_id < len(docs)
            ):
                raise ValueError(f"Trace item does not match fixed task: {row['id']}")
            source[row["id"]] = (doc_id, row["image_digests"])
    if not source:
        raise ValueError("Empty quality trace")

    responses = {}
    groups = set()
    with args.responses.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            request_id = row["request_id"]
            if (
                request_id not in source
                or request_id in responses
                or row.get("dataset_revision") != revision
                or (row.get("doc_id"), row["image_digests"]) != source[request_id]
            ):
                raise ValueError(f"Response does not match one unique trace item: {request_id}")
            responses[request_id] = row
            groups.add(row["group"])
    if len(groups) > 1:
        raise ValueError("Responses contain multiple experiment groups")

    filter_pipeline, = task._filters
    items = []
    for request_id, (doc_id, _) in source.items():
        doc = docs[doc_id]
        response = responses.get(request_id, {"status": "missing"})
        output = "".join(chunk[1] for chunk in response.get("content_chunks", []))
        item = {
            "request_id": request_id,
            "doc_id": doc_id,
            "status": response["status"],
            "raw_response": output,
            "correct": None,
        }
        if args.task == "muirbench":
            item.update(idx=doc["idx"], counterpart_idx=doc["counterpart_idx"])
        if response["status"] == "complete":
            instance = SimpleNamespace(resps=[output], filtered_resps={})
            filter_pipeline.apply([instance], [doc])
            value = instance.filtered_resps[filter_pipeline.name]
            metrics = task.process_results(doc, [value])
            if args.task == "ocrbench":
                correct = metrics["ocrbench_accuracy"]["score"]
            elif args.task == "muirbench":
                score = metrics["muirbench_score_overall"]
                correct = score["pred"].lower().strip() == score["answer"].lower().strip()
            else:
                correct = metrics["exact_match"]
            item.update(filtered_response=value, metrics=metrics, correct=bool(correct))
        items.append(item)

    completed = sum(item["status"] == "complete" for item in items)
    correct = sum(item["correct"] is True for item in items)
    report = {
        "task": args.task,
        "dataset_revision": revision,
        "group": next(iter(groups), None),
        "filter": filter_pipeline.name,
        "submitted": len(items),
        "completed": completed,
        "correct": correct,
        "accuracy_completed": correct / completed if completed else None,
        "service_accuracy": correct / len(items),
        "official_score": correct / len(docs) if completed == len(docs) else None,
        "items": items,
    }
    if args.task == "muirbench":
        by_id = {item["idx"]: item for item in items}
        pairs = {
            tuple(sorted((idx, item["counterpart_idx"])))
            for idx, item in by_id.items()
            if item["counterpart_idx"] in by_id and item["counterpart_idx"] != idx
        }
        scored = sum(all(by_id[idx]["correct"] is not None for idx in pair) for pair in pairs)
        both_correct = sum(all(by_id[idx]["correct"] is True for idx in pair) for pair in pairs)
        report.update(
            pairs_submitted=len(pairs),
            pairs_completed=scored,
            pairs_both_correct=both_correct,
            pair_accuracy_completed=both_correct / scored if scored else None,
            pair_service_accuracy=both_correct / len(pairs) if pairs else None,
        )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


async def replay(args):
    import aiohttp

    requests = []
    interference = getattr(args, "interference", None)
    arrival_scale = getattr(args, "arrival_scale", 1.0)
    heavy_trace = args.heavy_trace if interference in ("same", "separate") else None
    paths = [args.trace, heavy_trace] if heavy_trace else [args.trace]
    run_id = uuid.uuid4().hex[:12]
    seen_ids = set()
    with ExitStack() as resources:
        streams = [resources.enter_context(path.open(encoding="utf-8")) for path in paths]
        for index, lines in enumerate(streams):
            if interference:
                line = lines.readline()
                if not line or lines.readline():
                    raise ValueError(f"{paths[index]} must contain exactly one frozen request")
                streams[index] = [line]
        for index, line in enumerate(chain.from_iterable(streams)):
            row = json.loads(line)
            if row["id"] in seen_ids:
                raise ValueError(f"duplicate logical request ID: {row['id']}")
            seen_ids.add(row["id"])
            row["engine_request_id"] = f"{run_id}-{row['id']}"
            images = row.pop("images")
            if interference:
                if "task" in row or not row.get("ignore_eos", True) or row["at"] != 0:
                    raise ValueError("Interference requires fixed-output synthetic traces at offset zero")
                if index == 0 and row["output_tokens"] != 256:
                    raise ValueError("The long request must have a fixed 256-token output")
                if index == 1 and (len(images) != 8 or len(row["image_digests"]) != 8):
                    raise ValueError("The interference request must contain eight images")
            content = [
                {"type": "image_url", "image_url": {"url": data}} for data in images
            ]
            text = {"type": "text", "text": row.pop("text")}
            if "task" in row:
                content.insert(0, text)
            else:
                content.append(text)
            payload = json.dumps(
                {
                    "model": args.model,
                    "rid": row["engine_request_id"],
                    "messages": [{"role": "user", "content": content}],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "temperature": 0,
                    "ignore_eos": row.get("ignore_eos", True),
                    "max_tokens": row["output_tokens"],
                },
                ensure_ascii=False,
            ).encode("utf-8")
            requests.append((row, payload))
    loop = asyncio.get_running_loop()
    first_long_content = loop.create_future() if heavy_trace else None
    long_finished = asyncio.Event() if heavy_trace else None
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=args.timeout),
        connector=aiohttp.TCPConnector(limit=0),
        trust_env=False,
    ) as session:
        with args.output.open("x", encoding="utf-8", buffering=1) as output:
            start = loop.time()
            wall_origin = time.time_ns()

            async def request(index, row, payload):
                route_index = 0 if interference == "same" else index
                endpoint = args.endpoints[(route_index + args.route_offset) % len(args.endpoints)]
                planned = row["at"] * arrival_scale
                skip_reason = None
                trigger = None
                if interference and index == 1:
                    try:
                        trigger = await asyncio.wait_for(
                            asyncio.shield(first_long_content), timeout=args.timeout
                        )
                    except asyncio.TimeoutError:
                        skip_reason = "timed out waiting for first long-request content"
                    if trigger is None:
                        planned = None
                        skip_reason = skip_reason or "long request ended before first content"
                    else:
                        planned = trigger - start + args.heavy_delay_s
                        wait = planned + start - loop.time()
                        if wait > 0:
                            try:
                                await asyncio.wait_for(long_finished.wait(), timeout=wait)
                            except asyncio.TimeoutError:
                                pass
                        if long_finished.is_set():
                            skip_reason = "long request ended before the eight-image request was sent"
                else:
                    await asyncio.sleep(max(0, start + planned - loop.time()))
                sent = loop.time()
                record = {
                    "group": args.group,
                    "request_id": row["id"],
                    "engine_request_id": row["engine_request_id"],
                    "endpoint": endpoint,
                    "planned_offset_s": planned,
                    "send_offset_s": None if skip_reason else sent - start,
                    "wall_origin_ns": wall_origin,
                    "image_count": len(row["image_digests"]),
                    "image_digests": row["image_digests"],
                    "vision_tokens_estimate": row["vision_tokens_estimate"],
                    "expected_output_tokens": row["output_tokens"],
                    "client_http_body_bytes": 0 if skip_reason else len(payload),
                    "content_chunks": [],
                    "response_ids": [],
                    "completion_tokens": None,
                    "status": "failed",
                }
                if interference:
                    record["interference_mode"] = interference
                    record["interference_role"] = "heavy" if index == 1 else "long"
                    record["trigger_offset_s"] = trigger - start if trigger is not None else None
                if skip_reason:
                    record.update(status="skipped", error=skip_reason, duration_s=None)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    return record["status"]
                for key in ("task", "dataset_revision", "doc_id", "idx", "counterpart_idx"):
                    if key in row:
                        record[key] = row[key]
                try:
                    async with session.post(
                        endpoint.rstrip("/") + "/v1/chat/completions",
                        data=payload,
                        headers={
                            "Content-Type": "application/json",
                            "X-Request-ID": row["engine_request_id"],
                        },
                    ) as response:
                        record["http_status"] = response.status
                        response.raise_for_status()
                        async for line in response.content:
                            if not line.startswith(b"data:"):
                                continue
                            data = line[5:].strip()
                            if data == b"[DONE]":
                                record["status"] = "complete"
                                break
                            event = json.loads(data)
                            if event.get("error"):
                                raise RuntimeError(str(event["error"]))
                            rid = event.get("id")
                            if rid and rid not in record["response_ids"]:
                                record["response_ids"].append(rid)
                            usage = event.get("usage") or {}
                            if "completion_tokens" in usage:
                                record["completion_tokens"] = usage["completion_tokens"]
                            for choice in event.get("choices", []):
                                content_delta = choice.get("delta", {}).get("content")
                                if content_delta:
                                    now = loop.time()
                                    record["content_chunks"].append([now - sent, content_delta])
                                    if (
                                        index == 0
                                        and first_long_content is not None
                                        and not first_long_content.done()
                                    ):
                                        first_long_content.set_result(now)
                                if choice.get("finish_reason"):
                                    record["finish_reason"] = choice["finish_reason"]
                        if record["status"] != "complete":
                            raise EOFError("missing stream terminator")
                        if not record["content_chunks"]:
                            raise ValueError("no output content")
                        if row.get("ignore_eos", True) and (
                            record["completion_tokens"] != row["output_tokens"]
                        ):
                            raise ValueError("unexpected completion token count")
                except Exception as error:
                    record["status"] = "failed"
                    record["error"] = f"{type(error).__name__}: {error}"
                finally:
                    if index == 0 and first_long_content is not None:
                        if not first_long_content.done():
                            first_long_content.set_result(None)
                        long_finished.set()
                record["duration_s"] = loop.time() - sent
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                return record["status"]

            outcomes = await asyncio.gather(
                *(request(index, *pair) for index, pair in enumerate(requests))
            )
    skipped = outcomes.count("skipped")
    completed = outcomes.count("complete")
    totals = {"submitted": len(requests) - skipped, "completed": completed}
    if interference:
        totals["skipped"] = skipped
    print(json.dumps(totals))
    return 0 if completed == len(requests) else 1


def percentiles(values):
    ordered = sorted(values)
    result = {"n": len(ordered)}
    for point in (50, 95, 99):
        value = None
        if ordered:
            rank = (len(ordered) - 1) * point / 100
            lower = int(rank)
            upper = min(lower + 1, len(ordered) - 1)
            value = ordered[lower] + (rank - lower) * (ordered[upper] - ordered[lower])
        result[f"p{point}"] = value
    return result


def summarize_run(path, expected_requests):
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"No replay response records in {path}")
    origins = {row["wall_origin_ns"] for row in rows}
    groups = {row["group"] for row in rows}
    if len(origins) != 1 or len(groups) != 1:
        raise ValueError(f"{path} must contain exactly one replay")
    if len({row["request_id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate request ID in {path}")
    if expected_requests is not None and len(rows) > expected_requests:
        raise ValueError(f"{path} logged more than the expected number of requests")

    endpoints = defaultdict(
        lambda: {"submitted": 0, "completed": 0, "failed": 0, "images": 0, "completed_images": 0}
    )
    starts, finishes = [], []
    ttft, tpot, content_gaps = [], [], []
    lengths, expected_lengths, failures, skips = Counter(), Counter(), Counter(), Counter()
    counted_tokens = 0
    mismatched_lengths = 0
    for row in rows:
        endpoint = endpoints[row["endpoint"]]
        if row["status"] == "skipped":
            if row["interference_role"] != "heavy" or row["send_offset_s"] is not None:
                raise ValueError(f"Invalid skipped response for {row['request_id']} in {path}")
            skips[row["error"]] += 1
            endpoint["skipped"] = endpoint.get("skipped", 0) + 1
            continue
        send, duration = row["send_offset_s"], row["duration_s"]
        chunks = [entry[0] for entry in row["content_chunks"]]
        if (
            not all(math.isfinite(x) and x >= 0 for x in (send, duration, *chunks))
            or any(a > b for a, b in zip(chunks, chunks[1:]))
            or (chunks and chunks[-1] > duration)
        ):
            raise ValueError(f"Invalid replay timing for {row['request_id']} in {path}")
        starts.append(send)
        finishes.append(send + duration)
        endpoint["submitted"] += 1
        endpoint["images"] += row["image_count"]
        if row["status"] == "failed":
            endpoint["failed"] += 1
            failures[row.get("error", "Unknown error").partition(":")[0]] += 1
            continue
        if row["status"] != "complete" or not chunks:
            raise ValueError(f"Invalid completed response for {row['request_id']} in {path}")

        endpoint["completed"] += 1
        endpoint["completed_images"] += row["image_count"]
        ttft.append(chunks[0] * 1000)
        content_gaps.extend((b - a) * 1000 for a, b in zip(chunks, chunks[1:]))
        actual, expected = row["completion_tokens"], row["expected_output_tokens"]
        expected_lengths[expected] += 1
        if actual is not None:
            if not isinstance(actual, int) or actual < 0:
                raise ValueError(f"Invalid usage for {row['request_id']} in {path}")
            lengths[actual] += 1
            counted_tokens += actual
            mismatched_lengths += actual != expected
            if actual > 1:
                tpot.append((duration - chunks[0]) * 1000 / (actual - 1))

    submitted, completed = len(starts), len(ttft)
    observation = max(finishes) - min(starts) if starts else 0
    known_length_requests = sum(lengths.values())
    return {
        "response_log": str(path),
        "group": next(iter(groups)),
        "wall_origin_ns": next(iter(origins)),
        "submitted": submitted,
        "completed": completed,
        "failed": submitted - completed,
        "failure_rate": (submitted - completed) / submitted if submitted else None,
        "skipped": sum(skips.values()),
        "skips_by_reason": dict(skips),
        "expected_requests": expected_requests,
        "unlogged_requests": expected_requests - len(rows) if expected_requests is not None else None,
        "failures_by_type": dict(failures),
        "observation_s": observation,
        "completed_requests_per_s": completed / observation if observation else None,
        "ttft_ms": percentiles(ttft),
        "request_tpot_ms": percentiles(tpot),
        "sse_content_chunk_gap_ms": percentiles(content_gaps),
        "actual_output_tokens": {
            "known_requests": known_length_requests,
            "unknown_requests": completed - known_length_requests,
            "min": min(lengths, default=None),
            "max": max(lengths, default=None),
            "total": counted_tokens,
            "histogram": dict(sorted(lengths.items())),
            "different_from_requested": mismatched_lengths,
        },
        "requested_output_tokens_histogram": dict(sorted(expected_lengths.items())),
        "endpoints": dict(sorted(endpoints.items())),
    }


def summarize(args):
    report = {
        "observation": "First logged send to the last logged request end, including failures",
        "denominator": "Submitted counts logged HTTP attempts. Skipped records did not send an HTTP request.",
        "quantiles": "Linear interpolation over valid samples from completed requests",
        "request_tpot": "(request duration - TTFT) / (completion tokens - 1)",
        "sse_content_chunk_gap": "Adjacent nonempty SSE content chunks within each request",
        "runs": [summarize_run(path, args.expected_requests) for path in args.responses],
    }
    if args.output is None:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")


async def replay_matrix(args):
    """Replay the exact traces generated by the complete A matrix."""
    with (args.run_dir / "formal-design.json").open(encoding="utf-8") as stream:
        plan = json.load(stream)
    identifiers = [
        (f"formal-{images}img-{rate}qps-seed{seed}", plan["requests_per_cell"])
        for seed in plan["seeds"]
        for images, rates in plan["rates_qps"].items()
        for rate in rates
    ]
    official = plan["paper"]
    identifiers.extend(
        (
            f"official-{official['random_images_upper_bound']}img-{rate}qps-seed{official['seed']}",
            official["requests"],
        )
        for rate in official["target_rates"]
    )
    for name, count in identifiers:
        trace = args.run_dir / f"{name}.trace.jsonl"
        source = args.run_dir / f"{name}.A.jsonl"
        if not trace.is_file() or not source.is_file():
            raise FileNotFoundError(f"{name}: A trace or response log is missing")
        with source.open("rb") as stream:
            if sum(1 for _ in stream) != count:
                raise ValueError(f"{source}: incomplete A submission record")

    failures = 0
    for name, count in identifiers:
        suffix = f".{args.label}" if args.label else ""
        output = args.run_dir / f"{name}.{args.group}{suffix}.jsonl"
        if output.exists():
            with output.open("rb") as stream:
                if sum(1 for _ in stream) != count:
                    raise ValueError(f"{output}: partial original result requires review")
            print(f"FORMAL_CELL_RETAINED {name}", flush=True)
            continue
        print(f"FORMAL_CELL_BEGIN {name}", flush=True)
        case = argparse.Namespace(
            **vars(args),
            output=output,
            trace=args.run_dir / f"{name}.trace.jsonl",
            timeout=plan["timeout_per_request_s"],
            route_offset=0,
            interference=None,
            heavy_trace=None,
            heavy_delay_s=0,
        )
        status = await replay(case)
        print(f"FORMAL_CELL_END {name} status={status}", flush=True)
        failures += status
        if status:
            from urllib.request import urlopen

            for endpoint in args.endpoints:
                with urlopen(endpoint.rstrip("/") + "/health", timeout=4):
                    pass
    print(f"FORMAL_{args.group}_SYNTHETIC_COMPLETE failed_cells={failures}", flush=True)
    return int(failures > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    generate_args = actions.add_parser("generate")
    generate_args.add_argument("--trace", type=Path, required=True)
    generate_args.add_argument("--model", required=True)
    generate_args.add_argument("--requests", type=int, default=128)
    generate_args.add_argument("--images", type=int, required=True)
    generate_args.add_argument("--random-images", action="store_true")
    generate_args.add_argument("--output-tokens", type=int, default=64)
    generate_args.add_argument("--qps", type=float, required=True)
    generate_args.add_argument("--seed", type=int, required=True)
    quality_args = actions.add_parser("generate-quality")
    score_args = actions.add_parser("score-quality")
    for action in (quality_args, score_args):
        action.add_argument("--trace", type=Path, required=True)
        action.add_argument("--task", choices=QUALITY_DATASETS, required=True)
        action.add_argument("--lmms-eval", type=Path, required=True)
        action.add_argument(
            "--hub-cache", type=Path, default=Path.home() / ".cache/huggingface/hub"
        )
    quality_args.add_argument("--limit", type=int)
    quality_args.add_argument("--qps", type=float, required=True)
    quality_args.add_argument("--seed", type=int, required=True)
    score_args.add_argument("--responses", type=Path, required=True)
    score_args.add_argument("--output", type=Path, required=True)
    summary_args = actions.add_parser("summarize")
    summary_args.add_argument("responses", type=Path, nargs="+")
    summary_args.add_argument("--expected-requests", type=int)
    summary_args.add_argument("--output", type=Path)
    matrix_args = actions.add_parser("replay-matrix")
    matrix_args.add_argument("--run-dir", type=Path, required=True)
    matrix_args.add_argument("--model", required=True)
    matrix_args.add_argument("--group", choices=["B", "C70", "C50"], required=True)
    matrix_args.add_argument("--endpoints", nargs=1, required=True)
    matrix_args.add_argument("--label", help="Add a suffix while preserving the original results")
    run_args = actions.add_parser("replay")
    run_args.add_argument("--trace", type=Path, required=True)
    run_args.add_argument("--output", type=Path, required=True)
    run_args.add_argument("--model", required=True)
    run_args.add_argument(
        "--group", choices=["A", "B", "C100", "C70", "C50", "235B-4", "235B-5", "235B-6"],
        required=True,
    )
    run_args.add_argument("--endpoints", nargs="+", required=True)
    run_args.add_argument("--route-offset", type=int, default=0)
    run_args.add_argument("--timeout", type=float, default=150)
    run_args.add_argument(
        "--arrival-scale", type=float, default=1.0,
        help="Multiply the frozen arrival offsets without rewriting the trace.",
    )
    run_args.add_argument(
        "--interference", choices=("control", "same", "separate"),
        help="Use a fixed 256-token trace with no heavy request, or route an eight-image request after first content.",
    )
    run_args.add_argument("--heavy-trace", type=Path)
    run_args.add_argument(
        "--heavy-delay-s", type=float, default=0,
        help="Seconds from the first nonempty content block of the long request.",
    )
    args = parser.parse_args()
    if args.action == "generate":
        if min(args.requests, args.images, args.output_tokens, args.qps) <= 0:
            parser.error("generation parameters must be positive")
        generate(args)
        return 0
    if args.action == "generate-quality":
        if args.qps <= 0 or (args.limit is not None and args.limit <= 0):
            parser.error("generation parameters must be positive")
        generate_quality(args)
        return 0
    if args.action == "score-quality":
        score_quality(args)
        return 0
    if args.action == "summarize":
        if args.expected_requests is not None and args.expected_requests <= 0:
            parser.error("expected requests must be positive")
        summarize(args)
        return 0
    if args.action == "replay-matrix":
        if args.label is not None and not args.label.replace("-", "").isalnum():
            parser.error("matrix label must contain letters, numbers or hyphens")
        return asyncio.run(replay_matrix(args))
    if len(args.endpoints) != (2 if args.group == "A" else 1):
        parser.error("A requires two endpoints; the other groups require one")
    if not math.isfinite(args.arrival_scale) or args.arrival_scale <= 0:
        parser.error("arrival scale must be finite and positive")
    if args.interference in ("same", "separate"):
        if args.heavy_trace is None:
            parser.error("same and separate interference require --heavy-trace")
        if not math.isfinite(args.timeout) or not 0 < args.timeout:
            parser.error("interference requires a positive finite timeout")
        if not 0 <= args.heavy_delay_s <= args.timeout:
            parser.error("heavy delay must be between zero and the request timeout")
        if args.interference == "separate" and (
            len({endpoint.rstrip("/") for endpoint in args.endpoints}) != 2
        ):
            parser.error("separate interference requires two distinct A endpoints")
    elif args.heavy_trace is not None or args.heavy_delay_s != 0:
        parser.error("heavy trace and delay require same or separate interference")
    return asyncio.run(replay(args))


if __name__ == "__main__":
    raise SystemExit(main())
