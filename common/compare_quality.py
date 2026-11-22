"""Compare two score reports on the same frozen questions."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from random import Random


def unique(rows, field):
    result = {}
    for row in rows:
        key = row[field]
        if key in result:
            raise ValueError(f"Duplicate {field}: {key}")
        result[key] = row
    return result


def interval(clusters, seed, repeats):
    values = list(clusters.values())
    rng = Random(seed)
    samples = []
    for _ in range(repeats):
        numerator = denominator = 0
        for delta, count in rng.choices(values, k=len(values)):
            numerator += delta
            denominator += count
        samples.append(100 * numerator / denominator)
    samples.sort()
    return [samples[round((repeats - 1) * p)] for p in (0.025, 0.975)]


def summarize(clusters, seed, repeats):
    count = sum(n for _, n in clusters.values())
    return {
        "denominator": count,
        "clusters": len(clusters),
        "delta_pp": 100 * sum(delta for delta, _ in clusters.values()) / count,
        "ci95_pp": interval(clusters, seed, repeats),
    }


def compare(base, test, images, seed, repeats):
    for field in ("task", "dataset_revision", "filter"):
        if base[field] != test[field]:
            raise ValueError(f"Score reports differ in {field}")
    left, right = (unique(report["items"], "request_id") for report in (base, test))
    for report, items in ((base, left), (test, right)):
        if (
            len(items) != report["submitted"]
            or sum(row["correct"] is not None for row in items.values()) != report["completed"]
            or sum(row["correct"] is True for row in items.values()) != report["correct"]
        ):
            raise ValueError("Score report denominators do not match its items")
    if not left or left.keys() != right.keys():
        raise ValueError("Score reports do not contain the same nonempty question set")

    task = base["task"]
    cluster_unit = {"ocrbench": "image SHA-256", "realworldqa": "question", "muirbench": "question pair"}[task]
    if task == "ocrbench" and left.keys() != images.keys():
        raise ValueError("OCRBench needs image fingerprints for every scored question")
    clusters = defaultdict(lambda: [0, 0])
    pair_outcomes = defaultdict(list)
    transitions = defaultdict(int)
    for request_id, one in sorted(left.items()):
        two = right[request_id]
        if one["doc_id"] != two["doc_id"]:
            raise ValueError(f"Document ID differs for {request_id}")
        unit = request_id
        if task == "ocrbench":
            unit = images[request_id]
        if task == "muirbench":
            for field in ("idx", "counterpart_idx"):
                if one[field] != two[field]:
                    raise ValueError(f"MuirBench pair membership differs for {request_id}")
            unit = tuple(sorted((one["idx"], one["counterpart_idx"])))
            pair_outcomes[unit].append(one)
        a, b = one["correct"], two["correct"]
        transitions[str(a), str(b)] += 1
        clusters[unit][0] += int(b is True) - int(a is True)
        clusters[unit][1] += 1

    both_official = base["official_score"] is not None and test["official_score"] is not None
    answer = {
        "task": task,
        "dataset_revision": base["dataset_revision"],
        "base": {key: base[key] for key in ("group", "submitted", "completed", "correct", "official_score")},
        "test": {key: test[key] for key in ("group", "submitted", "completed", "correct", "official_score")},
        "resampling_unit": cluster_unit,
        "bootstrap_repeats": repeats,
        "seed": seed,
        "official_full_coverage": both_official,
        "submitted_accuracy": summarize(clusters, seed, repeats),
        "transitions_base_to_test": {f"{a}->{b}": n for (a, b), n in sorted(transitions.items())},
    }
    if task == "muirbench":
        pair_deltas = {}
        for pair, rows in pair_outcomes.items():
            if len(rows) != 2 or rows[0]["idx"] != rows[1]["counterpart_idx"] or rows[1]["idx"] != rows[0]["counterpart_idx"]:
                raise ValueError(f"Incomplete or nonreciprocal MuirBench pair: {pair}")
            a = all(row["correct"] is True for row in rows)
            b = all(right[row["request_id"]]["correct"] is True for row in rows)
            pair_deltas[pair] = (int(b) - int(a), 1)
        answer["submitted_both_correct"] = summarize(pair_deltas, seed, repeats)
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--ocr-responses", type=Path, nargs=2, metavar=("BASE_JSONL", "TEST_JSONL"))
    parser.add_argument("--repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=23011)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1000:
        parser.error("use at least 1000 bootstrap samples")
    base = json.loads(args.base.read_text(encoding="utf-8"))
    test = json.loads(args.test.read_text(encoding="utf-8"))
    images = {}
    if base["task"] == "ocrbench":
        if args.ocr_responses is None:
            parser.error("OCRBench requires both raw response logs")
        fingerprints = []
        for path in args.ocr_responses:
            with path.open(encoding="utf-8") as source:
                items = unique((json.loads(line) for line in source), "request_id")
            fingerprints.append({key: tuple(row["image_digests"]) for key, row in items.items()})
        images, test_images = fingerprints
        if images != test_images or any(len(fingerprint) != 1 for fingerprint in images.values()):
            raise ValueError("OCRBench response logs disagree on image provenance")
    result = compare(base, test, images, args.seed, args.repeats)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
