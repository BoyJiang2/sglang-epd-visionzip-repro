"""Join paired VisionZip probe payloads in both directions."""

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "C"))
from check_identity import sent_frames


def collect(ratio, response_path, encoder_path, pd_path):
    with response_path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    expected = [(mode, n) for mode in ("serial", "mixed") for n in (1, 4, 8)]
    if [(r["mode"], len(r["images"])) for r in records] != expected:
        raise ValueError(f"{response_path}: expected six serial and mixed probes")
    frames = sent_frames(records, encoder_path, ratio)
    http_trace = pd_path.read_text(encoding="utf-8")
    observed = []
    for record, (rows, feature, metadata) in zip(records, frames, strict=True):
        rid = re.escape(record["rid"])
        transfer = re.findall(
            rf"epd_encode_http req_id={rid}_local_part_0 part_idx=0 body_bytes=(\d+) "
            r"tcp_payload_sent=(\d+|unavailable) tcp_retrans_sent=(\d+|unavailable) http_status=(\d+)",
            http_trace,
        )
        if len(transfer) != 1:
            raise ValueError(f"{record['rid']}: missing or duplicate HTTP transfer")
        body, sent, retrans, status = transfer[0]
        body = int(body)
        sent = None if sent == "unavailable" else int(sent)
        retrans = None if retrans == "unavailable" else int(retrans)
        if status != "200" or not body or (sent is not None and sent < body):
            raise ValueError(f"{record['rid']}: inconsistent PD-to-E transfer counts")
        observed.append(dict(
            mode=record["mode"], request_id=record["id"], image_digests=record["images"],
            image_rows=rows, feature_bytes=feature, embedding_metadata_bytes=metadata,
            pd_to_e_http_body_bytes=body, pd_to_e_tcp_payload_bytes=sent,
            pd_to_e_tcp_retrans_bytes=retrans,
        ))
    return observed


def report(groups):
    result = {}
    reference = None
    for ratio, rows in groups.items():
        key = [(r["mode"], r["request_id"], r["image_digests"]) for r in rows]
        if reference is None:
            reference = key
        elif key != reference:
            raise ValueError("Transfer diagnostics did not replay the same images in the same modes")
        feature = sum(r["feature_bytes"] for r in rows)
        metadata = sum(r["embedding_metadata_bytes"] for r in rows)
        body = sum(r["pd_to_e_http_body_bytes"] for r in rows)
        tcp = [r["pd_to_e_tcp_payload_bytes"] for r in rows if r["pd_to_e_tcp_payload_bytes"] is not None]
        result[f"C{ratio}"] = dict(
            probes=rows, feature_bytes=feature, embedding_metadata_bytes=metadata,
            pd_to_e_http_body_bytes=body,
            two_way_application_bytes=feature + metadata + body,
            pd_to_e_tcp_payload_samples=len(tcp),
            pd_to_e_tcp_payload_bytes=sum(tcp) if len(tcp) == len(rows) else None,
        )
    for item in result.values():
        baseline = result["C100"]
        for key in ("feature_bytes", "two_way_application_bytes"):
            item[key + "_saved_fraction"] = 1 - item[key] / baseline[key]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for ratio in (100, 70, 50):
        parser.add_argument(
            f"--c{ratio}", type=Path, nargs=3, required=True,
            metavar=("RESPONSES", "ENCODER_LOG", "PD_LOG"),
        )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report({ratio: collect(ratio, *getattr(args, f"c{ratio}")) for ratio in (100, 70, 50)})
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
