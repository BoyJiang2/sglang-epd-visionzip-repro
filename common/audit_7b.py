"""Audit raw 7B run evidence before registering any 235B replay."""

import argparse
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
import json
import math
from pathlib import Path
import re
import sys

from compare_interference import analyze_mode
from epd_replay import QUALITY_DATASETS, summarize_run
from report_performance import fingerprint, load_cell

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "C"))
from check_identity import signature, verify_compression, verify_transfer


GROUPS = ("A", "B", "C70", "C50")
AUTHOR_GROUPS = ("author100", "author70", "author50")


def audit_check(evidence, name, check, *args):
    try:
        details = check(*args)
    except FileNotFoundError as error:
        evidence[name] = dict(state="missing", file=Path(error.filename).name)
        return None
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        evidence[name] = dict(state="invalid", reason=str(error))
        return None
    evidence[name] = dict(state="verified", **details)
    return details


def matrix_row(path, group, count, images, output_tokens, kind, reference):
    rows = load_cell(path, group, count)
    signatures = {rid: fingerprint(row) for rid, row in rows.items()}
    if reference is not None and signatures != reference:
        raise ValueError("Request IDs, images or planned arrival times differ")
    for row in rows.values():
        if not (1 <= row["image_count"] <= images
                and row["image_count"] == len(row["image_digests"])
                and row["expected_output_tokens"] == output_tokens
                and (kind != "formal" or row["image_count"] == images)):
            raise ValueError(f"{path.name}: workload differs from frozen configuration")
    summary = summarize_run(path, count)
    return dict(fingerprints=signatures, **{
        key: summary[key] for key in ("submitted", "completed", "failed", "failures_by_type")
    })


def matrix(run, plan, evidence, b_label):
    paper = plan["paper"]
    formal = [(int(images), rate, seed, plan["requests_per_cell"], plan["output_tokens"])
              for images, rates in plan["rates_qps"].items() for rate in rates for seed in plan["seeds"]]
    official = [(paper["random_images_upper_bound"], rate, paper["seed"], paper["requests"], paper["output_tokens"])
                for rate in paper["target_rates"]]
    if (tuple(plan["groups"]) != GROUPS or len(set(formal)) != len(formal) or len(formal) != 27
            or len(set(official)) != len(official) or len(official) != 5 or plan["requests_per_cell"] != 128
            or paper["requests"] != 64 or plan["output_tokens"] != 64 or paper["output_tokens"] != 256):
        raise ValueError("Unexpected frozen plan, expected 108 by 128 formal and 20 by 64 official cells")
    for kind, configurations in (("formal", formal), ("official", official)):
        for images, rate, seed, count, output_tokens in configurations:
            case = f"{kind}-{images}img-{rate:g}qps-seed{seed}"
            reference = None
            for group in GROUPS:
                suffix = f".{b_label}" if group == "B" and b_label else ""
                path = run / f"{case}.{group}{suffix}.jsonl"
                name = f"matrix/{group}/{case}"
                info = audit_check(evidence, name, matrix_row, path, group, count,
                                   images, output_tokens, kind, reference)
                if info is not None:
                    received = evidence[name].pop("fingerprints")
                    if reference is None:
                        reference = received


def score(run, task, group, stem, require_all, reference, scheduled):
    _, revision, expected = QUALITY_DATASETS[task]
    path = run / f"{stem}.scores.json"
    score_file = json.loads(path.read_text(encoding="utf-8"))
    with (run / f"{stem}.jsonl").open(encoding="utf-8") as source:
        rows = list(map(json.loads, source))
    items = score_file["items"]
    raw = {row["request_id"]: row for row in rows}
    by_id = {item["request_id"]: item for item in items}
    if (len(rows) != len(raw) or len(items) != len(by_id) or len(raw) != expected or raw.keys() != by_id.keys()
            or {item["doc_id"] for item in items} != set(range(expected))
            or (score_file["task"], score_file["group"], score_file["dataset_revision"], score_file["submitted"])
            != (task, group, revision, expected)):
        raise ValueError(f"{path.name}: incomplete or mismatched dataset coverage")
    completed = sum(row["status"] == "complete" for row in rows)
    correct = sum(item["correct"] is True for item in items)
    official = score_file["official_score"]
    provenance = {}
    arrivals = {} if group in GROUPS else None
    for rid, row in raw.items():
        item = by_id[rid]
        reply = "".join(chunk[1] for chunk in row.get("content_chunks", []))
        if (row["doc_id"] != item["doc_id"] or row["group"] != group or row["dataset_revision"] != revision
                or row["status"] != item["status"] or not row["image_digests"] or item["raw_response"] != reply
                or (type(item["correct"]) is bool) != (row["status"] == "complete")):
            raise ValueError(f"{path.name}: score does not match raw request {rid}")
        pair = (row["idx"], row["counterpart_idx"]) if task == "muirbench" else None
        if pair is not None and pair != (item["idx"], item["counterpart_idx"]):
            raise ValueError(f"{path.name}: MuirBench pair does not match raw request {rid}")
        provenance[rid] = (tuple(row["image_digests"]), pair)
        if arrivals is not None:
            arrivals[rid] = (row["planned_offset_s"], row["expected_output_tokens"])
    if task == "muirbench":
        pairs = dict(pair for _, pair in provenance.values())
        if (len(pairs) != expected or any(first == second or pairs.get(second) != first for first, second in pairs.items())
                or score_file["pairs_submitted"] != expected // 2):
            raise ValueError(f"{path.name}: expected a reciprocal partner for every MuirBench question")
    if (score_file["completed"] != completed or score_file["correct"] != correct
            or (require_all and completed != expected)
            or (completed != expected and official is not None)
            or (completed == expected and (official is None or not math.isclose(official, correct / expected)))
            or (reference is not None and reference != provenance)
            or (arrivals is not None and scheduled is not None and arrivals != scheduled)):
        raise ValueError(f"{path.name}: incorrect score denominator, coverage or image provenance")
    return dict(submitted=expected, completed=completed, correct=correct,
                official_score=official, fingerprints=provenance, arrivals=arrivals)


def quality(run, evidence, muir_label):
    for task in QUALITY_DATASETS:
        reference = None
        scheduled = None
        groups = GROUPS if task == "muirbench" else (*GROUPS, *AUTHOR_GROUPS)
        for group in groups:
            if group in AUTHOR_GROUPS:
                stem = f"{group}-{task}"
            else:
                tag = f"-{muir_label}" if task == "muirbench" else ""
                stem = f"quality-{task}{tag}.{group}"
            key = f"quality/{task}/{group}"
            result = audit_check(evidence, key, score, run, task, group, stem,
                                 task == "muirbench" or group in AUTHOR_GROUPS, reference, scheduled)
            if result is not None:
                received = evidence[key].pop("fingerprints")
                arrivals = evidence[key].pop("arrivals")
                if reference is None:
                    reference = received
                if scheduled is None and arrivals is not None:
                    scheduled = arrivals


def matched_transfer(run):
    def probes(group):
        label = "B.no-probe" if group == "B" else group
        path = run / f"identity.{label}.jsonl"
        rows = list(map(json.loads, path.read_text(encoding="utf-8").splitlines()))
        if ([(row["mode"], len(row["images"])) for row in rows]
                != [(mode, n) for mode in ("serial", "mixed") for n in (1, 4, 8)]
                or len({row["rid"] for row in rows}) != 6):
            raise ValueError(f"{path.name}: expected six unique probes")
        return rows

    baseline = probes("B")
    reference = probes("C100")
    baseline_signatures = [signature(row) for row in baseline]
    signatures = [signature(row) for row in reference]
    if [entry[:2] for entry in baseline_signatures] != [entry[:2] for entry in signatures]:
        raise ValueError("C100 prompt lengths or generated token IDs differ from B")
    details = {}
    with redirect_stdout(StringIO()):
        for ratio, actual in ((100, reference), (70, probes("C70")), (50, probes("C50"))):
            if any((a["mode"], a["id"], a["images"]) != (b["mode"], b["id"], b["images"])
                   for a, b in zip(actual, baseline, strict=True)):
                raise ValueError(f"C{ratio}: replay did not use the six identical probes")
            phase = "identity" if ratio == 100 else "payload"
            logs = [run / f"C{ratio}-{node}-{phase}.service.log" for node in ("qjy000", "qjy001")]
            frames = verify_transfer(actual, *logs, ratio)
            receiver = logs[1].read_text(encoding="utf-8")
            starts = []
            for row in actual:
                delta = re.findall(
                    rf"visionzip_receive req_id={re.escape(row['rid'])} [^\n]* mrope_delta=\[(-?\d+)\]",
                    receiver,
                )
                if len(delta) != 1:
                    raise ValueError(f"{row['rid']}: expected one M-RoPE decode delta")
                starts.append(signature(row)[0] + int(delta[0]))
            if ratio == 100:
                original_frames = frames
                expected_starts = starts
            else:
                verify_compression(actual, reference, frames, original_frames, ratio)
                if starts != expected_starts:
                    raise ValueError(f"C{ratio}: compressed prompt changed the first decode position")
            details[f"C{ratio}"] = dict(probes=len(frames), feature_bytes=sum(f[1] for f in frames),
                                        metadata_bytes=sum(f[2] for f in frames), first_decode_positions=starts)
    details["C100"]["first_logprob_deltas_from_B"] = [
        observed[2] - expected[2] for expected, observed in zip(baseline_signatures, signatures, strict=True)
    ]
    return details


def audit(run, b_label, muir_label):
    evidence = {}
    plan = json.loads((run / "formal-design.json").read_text(encoding="utf-8"))
    matrix(run, plan, evidence, b_label)
    quality(run, evidence, muir_label)
    audit_check(evidence, "identity/B-C", matched_transfer, run)
    for group in GROUPS:
        for mode in (("control", "same", "separate") if group == "A" else ("control", "same")):
            key = f"interference/{group}/{mode}"
            if audit_check(evidence, key, analyze_mode, run, group, mode, 10) is not None:
                evidence[key].pop("per_repeat")
    counts = Counter(entry["state"] for entry in evidence.values())
    return dict(raw_evidence_ready=counts.get("verified") == len(evidence), counts=dict(counts), evidence=evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--b-label", default="no-probe")
    parser.add_argument("--muir-label", default="1qps")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9.-]+", args.muir_label) or (args.b_label and not re.fullmatch(r"[a-zA-Z0-9.-]+", args.b_label)):
        parser.error("Invalid replay label")
    report = audit(args.run_dir, args.b_label, args.muir_label)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report["counts"]))
    return 0 if report["raw_evidence_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
