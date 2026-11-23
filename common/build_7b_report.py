"""Build the 7B figures and analyses from verified run evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


GROUPS = ("A", "B", "C70", "C50")
COMPRESSED = ("C70", "C50")
ENCODER_GROUPS = ("B", *COMPRESSED)
QUALITY = ("ocrbench", "realworldqa", "muirbench")


def build(run, output, b_label, muir_label, client_hosts):
    source = Path(__file__).resolve().parent
    output.mkdir()

    def execute(script, *arguments):
        subprocess.run([sys.executable, "-B", str(source / script), *map(str, arguments)], check=True)

    execute("audit_7b.py", "--run-dir", run, "--b-label", b_label,
            "--muir-label", muir_label, "--output", output / "raw-evidence.json")
    execute("report_performance.py", "--run-dir", run, "--b-label", b_label,
            "--output", output / "performance.json", "--figure", output / "performance.png")
    execute("compare_interference.py", "--run-dir", run, "--groups", *GROUPS,
            "--output", output / "interference.json", "--figure", output / "interference.png")

    def compare(task, baseline, treatment):
        def stem(group):
            if group.startswith("author"):
                return f"{group}-{task}"
            prefix = f"quality-muirbench-{muir_label}" if task == "muirbench" else f"quality-{task}"
            return f"{prefix}.{group}"

        base, test = stem(baseline), stem(treatment)
        ocr_inputs = ["--ocr-responses", run / f"{base}.jsonl", run / f"{test}.jsonl"] if task == "ocrbench" else []
        execute("compare_quality.py", "--base", run / f"{base}.scores.json",
                "--test", run / f"{test}.scores.json", *ocr_inputs,
                "--output", output / f"{task}.{baseline}-{treatment}.json")

    for task in QUALITY:
        compare(task, "A", "B")
        for group in COMPRESSED:
            compare(task, "B", group)
        if task != "muirbench":
            for group in ("author70", "author50"):
                compare(task, "author100", group)

    def role_logs(group, phase):
        return [run / f"{group}-{node}-{phase}.service.log" for node in ("qjy000", "qjy001")]

    probe_args = []
    for group in ("C100", *COMPRESSED):
        phase = "identity" if group == "C100" else "payload"
        probe_args.extend([f"--{group.lower()}", run / f"identity.{group}.jsonl", *role_logs(group, phase)])
    execute("diagnostics/report_probe_transport.py", *probe_args, "--output", output / "probe-transport.json")

    host_args = [argument for host in client_hosts for argument in ("--client-host", host)]
    muir_args, quality_args = [], []
    for group in GROUPS:
        stem = f"quality-muirbench-{muir_label}.{group}"
        muir_args.extend([f"--{group.lower()}", run / f"{stem}.jsonl", run / f"{stem}.scores.json"])
        if group in ENCODER_GROUPS:
            logs = role_logs(group, f"muir-{muir_label}")
            muir_args.extend(logs)
            quality_args.extend([f"--{group.lower()}-logs", *logs])
    execute("diagnostics/report_muir_traffic.py", *host_args, *muir_args,
            "--output", output / "muir-traffic.json")
    execute("diagnostics/report_quality_probes.py", "--run-dir", run, *host_args, *quality_args,
            "--output", output / "quality-traffic.json", "--figure", output / "quality-traffic.png")
    stage_logs = []
    for group in ENCODER_GROUPS:
        encoder, pd = role_logs(group, "stage-timing")
        stage_logs.extend(["--encoder-log", group, encoder, "--pd-log", group, pd])
    execute("diagnostics/report_stage_timing.py", run / "stage-timing-manifest.json", *stage_logs,
            "--output", output / "stage-timing.json")

    artifacts = {
        file.name: dict(bytes=file.stat().st_size, sha256=hashlib.sha256(file.read_bytes()).hexdigest())
        for file in sorted(output.iterdir())
    }
    with (output / "analysis-receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(b_label=b_label, muir_label=muir_label, artifacts=artifacts), stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--b-label", default="no-probe")
    parser.add_argument("--muir-label", default="1qps")
    parser.add_argument("--client-host", required=True, action="append")
    args = parser.parse_args()
    try:
        build(args.run_dir, args.output_dir, args.b_label, args.muir_label, args.client_host)
    except subprocess.CalledProcessError as error:
        parser.exit(error.returncode, f"Analysis stopped at {Path(error.cmd[2]).name}\n")


if __name__ == "__main__":
    main()
