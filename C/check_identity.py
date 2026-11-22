"""Capture greedy and transfer probes outside of timed runs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import re
from urllib.request import Request, urlopen
from uuid import uuid4


def probe(endpoint, model, row, mode):
    rid = f"identity-{uuid4().hex}"
    content = [
        {"type": "image_url", "image_url": {"url": url}} for url in row["images"]
    ]
    payload = {
        "model": model,
        "rid": rid,
        "messages": [
            {"role": "user", "content": content + [{"type": "text", "text": row["text"]}]}
        ],
        "max_tokens": 16,
        "ignore_eos": True,
        "temperature": 0,
        "logprobs": True,
        "return_output_ids_in_sglext": True,
    }
    request = Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        json.dumps(payload).encode("utf-8"),
        {"Content-Type": "application/json"},
    )
    record = {"mode": mode, "id": row["id"], "rid": rid, "images": row["image_digests"]}
    try:
        with urlopen(request, timeout=180) as response:
            record["response"] = json.load(response)
    except Exception as error:
        record["error"] = str(error)
    return record


def signature(record):
    response = record.get("response", {})
    extension = response.get("sglext") or {}
    prompt_length = (response.get("usage") or {}).get("prompt_tokens")
    output_ids = extension.get("output_ids")
    content = response.get("choices", [{}])[0].get("logprobs") or {}
    logprobs = content.get("content") or []
    if (
        not isinstance(prompt_length, int)
        or prompt_length <= 0
        or not output_ids
        or len(output_ids[0]) != 16
        or len(logprobs) != 16
    ):
        raise ValueError(f"{record['rid']}: incomplete token evidence")
    first_logprob = logprobs[0]["logprob"]
    if not math.isfinite(first_logprob):
        raise ValueError(f"{record['rid']}: nonfinite first-token log probability")
    return prompt_length, output_ids, first_logprob


def sent_frames(records, encoder_log, ratio):
    encoder = encoder_log.read_text(encoding="utf-8")
    frames = []
    for record in records:
        rid = re.escape(record["rid"])
        parts = re.findall(
            rf"visionzip_payload req_id={rid}_local_part_0 ratio={ratio} rows=(\d+) "
            r"feature_bytes=(\d+) metadata_bytes=(\d+) image_rows=(\[[^\n]*?\])",
            encoder,
        )
        if len(parts) != 1:
            raise ValueError(f"{record['rid']}: missing or duplicate encoder frame")
        row_count, feature_bytes, metadata_bytes = map(int, parts[0][:3])
        image_rows = json.loads(parts[0][3])
        if (
            len(image_rows) != len(record["images"])
            or any(n <= 0 for n in image_rows)
            or sum(image_rows) != row_count
            or min(feature_bytes, metadata_bytes) <= 0
        ):
            raise ValueError(f"{record['rid']}: invalid encoder frame")
        frames.append((image_rows, feature_bytes, metadata_bytes))
    return frames


def verify_transfer(records, encoder_log, pd_log, ratio=100):
    frames = sent_frames(records, encoder_log, ratio)
    receiver = pd_log.read_text(encoding="utf-8")
    for record, (image_rows, feature_bytes, metadata_bytes) in zip(records, frames, strict=True):
        rid = re.escape(record["rid"])
        received_layouts = re.findall(
            rf"visionzip_receive req_id={rid} image_rows=(\[[^\n]*?\]) "
            r"placeholders=(\[[^\n]*?\]) prompt_length=(\d+) mrope_length=(\d+)",
            receiver,
        )
        if len(received_layouts) != 1:
            raise ValueError(f"{record['rid']}: missing or duplicate PD layout")
        received, = received_layouts
        prompt_len, position_len = map(int, received[2:])
        if not (
            image_rows == json.loads(received[0]) == json.loads(received[1])
            and prompt_len == position_len == signature(record)[0]
        ):
            raise ValueError(f"{record['rid']}: C{ratio} payload and PD layout disagree")
        print(f"{record['rid']} image_rows={image_rows} feature_bytes={feature_bytes} metadata_bytes={metadata_bytes}")
    return frames


def verify_compression(actual, reference, compressed, full, ratio):
    for row, baseline, compact_frame, full_frame in zip(actual, reference, compressed, full, strict=True):
        rid = row["rid"]
        compact_rows, compact_bytes, _ = compact_frame
        full_rows, full_bytes, _ = full_frame
        expected = [int((ratio - 5) * n / 100) + max(int(0.05 * n), 1) for n in full_rows]
        if compact_rows != expected:
            raise ValueError(f"{rid}: C{ratio} keep counts {compact_rows}, expected {expected}")
        if compact_bytes * sum(full_rows) != full_bytes * sum(compact_rows):
            raise ValueError(f"{rid}: C{ratio} changed bytes per feature row")
        compact_prompt, _, _ = signature(row)
        full_prompt, _, _ = signature(baseline)
        if full_prompt - compact_prompt != sum(full_rows) - sum(compact_rows):
            raise ValueError(f"{rid}: C{ratio} shortened the prompt by an unexpected amount")
        print(f"{row['mode']} {row['id']}: feature bytes {full_bytes} → {compact_bytes}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--traces", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--ratio", type=int, choices=(100, 70, 50), default=100)
    parser.add_argument("--reference-encoder-log", type=Path)
    parser.add_argument("--transfer-logs", nargs=2, type=Path, metavar=("ENCODER_LOG", "PD_LOG"))
    args = parser.parse_args()
    if args.ratio != 100 and not (args.reference and args.reference_encoder_log and args.transfer_logs):
        parser.error("compressed probes require a C100 reference and both groups' encoder logs")
    rows = []
    for path in args.traces:
        with path.open(encoding="utf-8") as trace:
            rows.append(json.loads(next(trace)))
    if [len(row["images"]) for row in rows] != [1, 4, 8]:
        parser.error("expected one-, four-, and eight-image traces in order")
    with args.output.open("x", encoding="utf-8", buffering=1) as output:
        for row in rows:
            print(json.dumps(probe(args.endpoint, args.model, row, "serial")), file=output)
        with ThreadPoolExecutor(max_workers=3) as pool:
            for result in pool.map(
                lambda row: probe(args.endpoint, args.model, row, "mixed"), rows
            ):
                print(json.dumps(result), file=output)
    actual = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines()]
    actual_signatures = [signature(record) for record in actual]
    if args.reference:
        reference = [json.loads(line) for line in args.reference.read_text(encoding="utf-8").splitlines()]
        for observed, expected, observed_sig in zip(actual, reference, actual_signatures, strict=True):
            expected_sig = signature(expected)
            if any(observed[key] != expected[key] for key in ("mode", "id", "images")):
                raise ValueError("The two runs did not replay the same probes")
            if args.ratio == 100:
                if observed_sig[:2] != expected_sig[:2]:
                    raise ValueError(f"{observed['mode']} {observed['id']}: prompt length or generated token ID mismatch")
                print(
                    f"{observed['mode']} {observed['id']}: matched prompt length and generated token IDs, "
                    f"first-logprob delta={observed_sig[2] - expected_sig[2]:.8g}"
                )
    if args.transfer_logs:
        compressed = verify_transfer(actual, *args.transfer_logs, args.ratio)
        if args.ratio != 100:
            full = sent_frames(reference, args.reference_encoder_log, 100)
            verify_compression(actual, reference, compressed, full, args.ratio)
    print(f"{len(actual)} probes captured")


if __name__ == "__main__":
    main()
