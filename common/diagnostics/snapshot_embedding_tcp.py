"""Read Linux TCP_INFO for one experiment-owned persistent E-to-PD socket."""

import argparse
import ipaddress
import json
import os
from pathlib import Path
import re
import subprocess
import time


def snapshot(pid, peer, port):
    process = Path("/proc") / str(pid)
    if process.stat().st_uid != os.getuid():
        raise ValueError("The embedding process must belong to the current user")
    start_ticks = process.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]
    listing = subprocess.check_output(
        ["ss", "-4", "-Htinpe", "state", "established", "dst", peer, "dport", "=", f":{port}"],
        text=True, timeout=5,
    )
    sockets = [
        s for s in re.split(r"\n(?=\S)", listing.strip())
        if re.search(rf"\bpid={pid},fd=\d+\b", s.partition("\n")[0])
    ]
    if len(sockets) != 1:
        raise ValueError(f"Expected one embedding socket for pid {pid}, got {len(sockets)}")
    raw, = sockets
    header, _, stats = raw.partition("\n")
    _, send_queue, local, remote = header.split(maxsplit=4)[:4]
    identity = {}
    for field in ("ino", "sk"):
        match = re.search(rf"\b{field}:([0-9a-f]+)\b", header)
        if match is None:
            raise ValueError(f"Missing socket identity field {field}")
        identity[field] = match[1]
    counters = {}
    for field in ("bytes_sent", "bytes_acked"):
        match = re.search(rf"\b{field}:(\d+)\b", stats)
        if match is None:
            raise ValueError(f"Missing {field}. Warm up the socket before measuring.")
        counters[field] = int(match[1])
    if process.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19] != start_ticks:
        raise ValueError("Process changed during the snapshot")
    return {
        "identity": dict(identity, pid=pid, start_ticks=int(start_ticks), local=local, remote=remote,
                         boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip()),
        "epoch_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
        "send_queue_bytes": int(send_queue), **counters, "ss_output": raw,
    }


def difference(before, after):
    if before["identity"] != after["identity"] or before["monotonic_ns"] >= after["monotonic_ns"]:
        raise ValueError("Snapshots must be ordered samples of the same socket")
    sent = after["bytes_sent"] - before["bytes_sent"]
    acked = after["bytes_acked"] - before["bytes_acked"]
    if min(sent, acked) < 0:
        raise ValueError("A TCP counter decreased")
    return {
        "tcp_payload_bytes_sent": sent,
        "tcp_payload_bytes_acked": acked,
        "both_send_queues_empty": before["send_queue_bytes"] == after["send_queue_bytes"] == 0,
        "elapsed_s": (after["monotonic_ns"] - before["monotonic_ns"]) / 1e9,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    read = commands.add_parser("snapshot")
    read.add_argument("--pid", required=True, type=int)
    read.add_argument("--peer", required=True, type=ipaddress.IPv4Address)
    read.add_argument("--port", required=True, type=int)
    read.add_argument("--output", required=True, type=Path)
    delta = commands.add_parser("delta")
    delta.add_argument("before", type=Path)
    delta.add_argument("after", type=Path)
    args = parser.parse_args()
    if args.command == "snapshot":
        if args.pid <= 0 or not 1 <= args.port <= 65535:
            parser.error("expected a positive PID and a valid peer port")
        result = snapshot(args.pid, str(args.peer), args.port)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    else:
        before, after = (json.loads(path.read_text(encoding="utf-8")) for path in (args.before, args.after))
        print(json.dumps(difference(before, after), indent=2))


if __name__ == "__main__":
    main()
