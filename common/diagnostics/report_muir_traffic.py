"""Join MuirBench scores, client latency, and observed cross-node traffic."""

import argparse
from collections import defaultdict
from ipaddress import ip_address
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit


REVISION = "4c393cffc985c77d28de3b9045e2e5186920df80"
ROWS = 2600
PD_TRACE = re.compile(
    r"epd_encode_http req_id=(?P<rid>\S+)_local_part_0 part_idx=0 "
    r"body_bytes=(?P<body>\d+) tcp_payload_sent=(?P<tcp>\d+|unavailable) "
    r"tcp_retrans_sent=(?P<retrans>\d+|unavailable) http_status=(?P<status>\S+)"
)
NATIVE_TRACE = re.compile(
    r"epd_feature_payload req_id=(?P<rid>\S+) rows=(?P<rows>\d+) "
    r"feature_bytes=(?P<feature>\d+) metadata_bytes=(?P<metadata>\d+)"
)
VISIONZIP_TRACE = re.compile(
    r"visionzip_payload req_id=(?P<rid>\S+) ratio=(?P<ratio>\d+) rows=(?P<rows>\d+) "
    r"feature_bytes=(?P<feature>\d+) metadata_bytes=(?P<metadata>\d+) "
    r"image_rows=(?P<image_rows>\[[^\]]*\])"
)


def index_log(path, pattern, engine_ids):
    records = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            match = pattern.search(line)
            if match:
                fields = match.groupdict()
                rid = fields.pop("rid").removesuffix("_local_part_0")
                if rid in engine_ids:
                    records[rid].append(fields)
    return records


def positive_int(value):
    return value if type(value) is int and value > 0 else None


def local_endpoint(endpoint, client_hosts):
    host = urlsplit(endpoint).hostname
    if not host:
        raise ValueError(f"Invalid endpoint: {endpoint}")
    host = host.casefold().rstrip(".")
    if host in client_hosts or host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def traffic(group, row, client_hosts, encoder, pd):
    complete = row["status"] == "complete"
    is_local = local_endpoint(row["endpoint"], client_hosts)
    body = row.get("client_http_body_bytes")
    sse = row.get("client_sse_bytes_observed")
    outbound = inbound = None
    result = {
        "client_http_body_bytes_prepared": body,
        "client_sse_bytes_observed": sse,
        "client_endpoint_local": is_local,
    }
    if group == "A":
        if complete:
            outbound = 0 if is_local else positive_int(body)
            inbound = 0 if is_local else positive_int(sse)
    else:
        if not is_local:
            raise ValueError(f"{row['request_id']}: PD endpoint must be local to the client")
        encoder_matches = encoder.get(row["engine_request_id"], [])
        pd_matches = pd.get(row["engine_request_id"], [])
        feature = metadata = rows = image_rows = http_body = tcp = retrans = status = None
        if len(encoder_matches) == 1:
            match = encoder_matches[0]
            rows, feature, metadata = (int(match[key]) for key in ("rows", "feature", "metadata"))
            if min(rows, feature, metadata) <= 0:
                raise ValueError(f"{row['request_id']}: empty feature payload")
            if group != "B":
                image_rows = json.loads(match["image_rows"])
                if (
                    int(match["ratio"]) != int(group[1:])
                    or len(image_rows) != row["image_count"]
                    or any(positive_int(count) is None for count in image_rows)
                    or sum(image_rows) != rows
                ):
                    raise ValueError(f"{row['request_id']}: inconsistent VisionZip payload")
            inbound = feature + metadata
        if len(pd_matches) == 1:
            match = pd_matches[0]
            http_body = int(match["body"])
            tcp = None if match["tcp"] == "unavailable" else int(match["tcp"])
            retrans = None if match["retrans"] == "unavailable" else int(match["retrans"])
            status = match["status"]
            if status == "200":
                outbound = positive_int(http_body)
                if tcp is not None and tcp < http_body:
                    raise ValueError(f"{row['request_id']}: TCP payload is smaller than HTTP body")
        result.update(
            encoder_log_matches=len(encoder_matches),
            pd_log_matches=len(pd_matches),
            encoder_output_rows=rows,
            encoder_image_rows=image_rows,
            e_to_pd_feature_bytes=feature,
            e_to_pd_metadata_bytes=metadata,
            pd_to_e_http_status=status,
            pd_to_e_http_body_bytes_observed=http_body,
            pd_to_e_tcp_payload_bytes=tcp,
            pd_to_e_tcp_retrans_bytes=retrans,
        )
    result.update(
        cross_node_outbound_application_bytes=outbound,
        cross_node_inbound_application_bytes=inbound,
        cross_node_total_application_bytes=(
            outbound + inbound if complete and outbound is not None and inbound is not None else None
        ),
    )
    return result


def collect(group, paths, client_hosts):
    response_path, score_path, *logs = paths
    with score_path.open(encoding="utf-8") as stream:
        scores = json.load(stream)
    items = scores["items"]
    by_idx = {item["idx"]: item["counterpart_idx"] for item in items}
    if (
        scores["task"] != "muirbench"
        or scores["dataset_revision"] != REVISION
        or scores["submitted"] != ROWS
        or len(items) != ROWS
        or {item["doc_id"] for item in items} != set(range(ROWS))
        or len(by_idx) != ROWS
        or any(idx == other or by_idx.get(other) != idx for idx, other in by_idx.items())
    ):
        raise ValueError(f"{score_path}: expected the pinned full MuirBench test set and reciprocal pairs")
    with response_path.open(encoding="utf-8") as stream:
        responses = {}
        engine_ids = set()
        for line in stream:
            row = json.loads(line)
            if row["request_id"] in responses or row["engine_request_id"] in engine_ids:
                raise ValueError(f"{response_path}: duplicate response identity")
            responses[row["request_id"]] = row
            engine_ids.add(row["engine_request_id"])
    if scores["group"] != (group if responses else None):
        raise ValueError(f"{score_path}: score group does not match {group}")
    encoder, pd = {}, {}
    if logs:
        encoder = index_log(logs[0], NATIVE_TRACE if group == "B" else VISIONZIP_TRACE, engine_ids)
        pd = index_log(logs[1], PD_TRACE, engine_ids)

    joined = []
    for item in items:
        doc_id = item["doc_id"]
        rid = item["request_id"]
        row = responses.pop(rid, None)
        status = item["status"]
        complete = status == "complete"
        if (
            rid != f"muirbench-{doc_id:05d}"
            or status != (row["status"] if row else "missing")
            or (type(item["correct"]) is not bool if complete else item["correct"] is not None)
        ):
            raise ValueError(f"{rid}: inconsistent score identity or status")
        metric = item.get("metrics", {}).get("muirbench_score_overall", {})
        if metric and metric["idx"] != item["idx"]:
            raise ValueError(f"{rid}: score metric has a different MuirBench index")
        chunks = row["content_chunks"] if row else []
        digest_order = row["image_digests"] if row else None
        if row and (
            (row["group"], row["task"], row["dataset_revision"]) != (group, "muirbench", REVISION)
            or any(row[key] != item[key] for key in ("doc_id", "idx", "counterpart_idx"))
            or len(digest_order) != row["image_count"]
            or "".join(chunk[1] for chunk in chunks) != item["raw_response"]
        ):
            raise ValueError(f"{rid}: response and score do not identify the same submission")
        ttft = chunks[0][0] if chunks else None
        if (complete and ttft is None) or (ttft is not None and (not math.isfinite(ttft) or ttft < 0)):
            raise ValueError(f"{rid}: invalid time to first content")
        record = {
            "mode": group,
            "request_id": rid,
            "engine_request_id": row["engine_request_id"] if row else None,
            "doc_id": doc_id,
            "idx": item["idx"],
            "counterpart_idx": item["counterpart_idx"],
            "task_category": metric.get("task"),
            "image_relation": metric.get("image_relation"),
            "image_type": metric.get("image_type"),
            "status": status,
            "correct": item["correct"],
            "ttft_ms": ttft * 1000 if ttft is not None else None,
            "endpoint": row["endpoint"] if row else None,
            "image_count": row["image_count"] if row else None,
            "image_digests": digest_order,
        }
        if row:
            record.update(traffic(group, row, client_hosts, encoder, pd))
        else:
            record.update(
                cross_node_outbound_application_bytes=None,
                cross_node_inbound_application_bytes=None,
                cross_node_total_application_bytes=None,
            )
        joined.append(record)
    if responses or (scores["completed"], scores["correct"]) != (
        sum(row["status"] == "complete" for row in joined),
        sum(row["correct"] is True for row in joined),
    ):
        raise ValueError(f"{response_path}: response and score counts differ")
    return joined


def report(groups, client_hosts):
    result = {
        "dataset_revision": REVISION,
        "test_rows": ROWS,
        "application_byte_scope": {
            "A": "Remote replica HTTP request body outbound, client-observed SSE through [DONE] inbound.",
            "B/C": "PD-to-E HTTP request body outbound, E-to-PD feature and metadata frames inbound.",
            "total": "Completed response and both directions known. Null otherwise.",
        },
        "groups": {},
    }
    identity, image_digests = {}, {}
    for group, paths in groups.items():
        rows = collect(group, paths, client_hosts)
        for row in rows:
            rid = row["request_id"]
            key = (row["doc_id"], row["idx"], row["counterpart_idx"])
            digests = row["image_digests"]
            if key != identity.setdefault(rid, key) or (
                digests is not None and digests != image_digests.setdefault(rid, digests)
            ):
                raise ValueError(f"{rid}: groups used different question pairs or image order")
        result["groups"][group] = rows
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--client-host", required=True, action="append",
        help="Client host or IP as used by a local endpoint. Repeat for aliases.",
    )
    parser.add_argument("--a", type=Path, nargs=2, metavar=("RESPONSES", "SCORES"))
    for group in ("b", "c70", "c50"):
        parser.add_argument(
            f"--{group}", type=Path, nargs=4, metavar=("RESPONSES", "SCORES", "ENCODER_LOG", "PD_LOG")
        )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    groups = {group.upper(): getattr(args, group) for group in ("a", "b", "c70", "c50") if getattr(args, group)}
    if not groups:
        parser.error("provide at least one experiment group")
    client_hosts = {host.casefold().rstrip(".") for host in args.client_host}
    joined = report(groups, client_hosts)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(joined, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
