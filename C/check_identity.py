"""Capture the same greedy B and C100 probes outside of timed runs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
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
        "return_input_ids_in_sglext": True,
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
    prompt_ids = extension.get("input_ids")
    output_ids = extension.get("output_ids")
    content = response.get("choices", [{}])[0].get("logprobs") or {}
    logprobs = content.get("content") or []
    if not prompt_ids or not output_ids or len(output_ids[0]) != 16 or len(logprobs) != 16:
        raise ValueError(f"{record['rid']}: incomplete token evidence")
    first_logprob = logprobs[0]["logprob"]
    if not math.isfinite(first_logprob):
        raise ValueError(f"{record['rid']}: nonfinite first-token log probability")
    return prompt_ids, output_ids, first_logprob


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--traces", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
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
                raise ValueError("B and C100 did not replay the same probes")
            if observed_sig[:2] != expected_sig[:2]:
                raise ValueError(f"{observed['mode']} {observed['id']}: token ID mismatch")
            print(
                f"{observed['mode']} {observed['id']}: matched token IDs, "
                f"first-logprob delta={observed_sig[2] - expected_sig[2]:.8g}"
            )
    print(f"{len(actual)} identity probes captured")


if __name__ == "__main__":
    main()
