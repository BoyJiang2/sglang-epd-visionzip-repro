# Shared replay and scoring

Run the commands from the repository root in the experiment Python environment. Synthetic generation uses the pinned SGLang checkout. Replay requires `aiohttp`. Quality trace generation and scoring use the pinned [lmms-eval checkout](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/d38f640d302911fcd3e83bf2f99f7dbac4ab9b89) and its Python dependencies.

## Synthetic workload

Set `MODEL_PATH` to the checkpoint directory and `MODEL_ID` to the server's advertised model name. Set `RUN_DIR` to a writable directory for experiment artifacts. Set `A_ENDPOINT_1`, `A_ENDPOINT_2`, and `PD_ENDPOINT` to the respective service origins.

```bash
mkdir -p "$RUN_DIR"
python3 common/epd_replay.py generate \
  --model "$MODEL_PATH" --trace "$RUN_DIR/synthetic.trace.jsonl" \
  --requests 128 --images 4 --output-tokens 64 --qps 2 --seed 42

python3 common/epd_replay.py replay \
  --model "$MODEL_ID" --trace "$RUN_DIR/synthetic.trace.jsonl" \
  --group A --endpoints "$A_ENDPOINT_1" "$A_ENDPOINT_2" \
  --output "$RUN_DIR/synthetic.A.jsonl"

python3 common/epd_replay.py replay \
  --model "$MODEL_ID" --trace "$RUN_DIR/synthetic.trace.jsonl" \
  --group B --endpoints "$PD_ENDPOINT" --output "$RUN_DIR/synthetic.B.jsonl"

python3 common/epd_replay.py summarize \
  "$RUN_DIR/synthetic.A.jsonl" "$RUN_DIR/synthetic.B.jsonl" \
  --expected-requests 128 --output "$RUN_DIR/synthetic.summary.json"
```

Reuse the trace for C70 and C50 with their respective PD endpoints. The summary reports TTFT, per-request TPOT, SSE content-block gaps, throughput, request outcomes, and endpoint distribution. For B, C70, and C50, `replay-matrix` reads `formal-design.json` and the associated A traces and response logs. Pass `--label no-probe` to write rerun results with a `.no-probe.jsonl` suffix and keep the original results. Individual runs support `--interference control|same|separate` for fixed 256-token streams. The two interference modes require a separate one-request, eight-image trace via `--heavy-trace`.

After all four groups finish the frozen matrix and the random-image sweeps, generate the per-run table and the common-load plot with Matplotlib installed.

```bash
python3 common/report_performance.py --run-dir "$RUN_DIR" --b-label no-probe \
  --output "$RUN_DIR/7b-performance.json" --figure "$RUN_DIR/7b-ttft.pdf"
```

The report requires one full request record per planned submission and checks IDs, image fingerprints, planned arrival offsets and requested output lengths across groups. The plot pools the three replicates at the QPS shared by all image counts (0.5 in the frozen plan). P95 TTFT uses completed requests, and the adjacent completion panel includes every submitted request.

Each cell reports failures before and after arrival of the first nonempty content block. For A→B, B→C70, and B→C50, the JSON also records both completed, baseline only, treatment only, and neither completed for paired request IDs. TTFT deltas subtract baseline from treatment on pairs that complete in both runs.

The [TPOT supplement](../plans/tail-p99.json) adds five 128-request repeats at the same single-image 4 QPS setting as the three formal repeats. Generate each new seed with the same synthetic generator and name its traces `tail-p99-1img-4qps-seed<seed>.trace.jsonl`. Replay each seed for all four groups and store `<trace-stem>.<group>.jsonl` beside the original evidence. Run `python3 common/report_tail_p99.py --run-dir "$RUN_DIR" --plan plans/tail-p99.json --output "$RUN_DIR/tail-p99-summary.json"` to verify the pairing and summarize the eight repeats. The 95% P99 interval resamples whole requests within each seed and requires at least 1,000 completed streams with 64 output tokens. SSE gap counts are reported separately.

For the long-generation interference runs, report adjacent SSE content-block gaps at the actual send time of the eight-image request. In the no-interference control use the first content block plus 50 ms. Include a gap straddling the alignment point so the first stall is visible. The report keeps attempted and completed repeats, the number of observed gaps, and 95% intervals for pooled post-trigger P95 and P99. Each bootstrap draw resamples whole long requests within a condition. An SSE block may contain more than one token.

```bash
python3 common/compare_interference.py --run-dir "$RUN_DIR" --groups A B C70 C50 \
  --output "$RUN_DIR/7b-interference.json" --figure "$RUN_DIR/interference.png"
```

## Data revisions

| Task | Hub dataset | Revision | Test rows |
|---|---|---|---:|
| `ocrbench` | [echo840/OCRBench](https://huggingface.co/datasets/echo840/OCRBench) | `92a54bd1384387c178d5a07140a2d85e0a3d12e1` | 1,000 |
| `realworldqa` | [lmms-lab/RealWorldQA](https://huggingface.co/datasets/lmms-lab/RealWorldQA) | `907c4e5228fd1703c710ed937601cb5f89ab8d5c` | 765 |
| `muirbench` | [MUIRBENCH/MUIRBENCH](https://huggingface.co/datasets/MUIRBENCH/MUIRBENCH) | `4c393cffc985c77d28de3b9045e2e5186920df80` | 2,600 |

The RealWorldQA revision follows the pinned lmms-eval task. The separate xAI publication has different row order and an updated question and label. MuirBench pairs are joined using the published `idx` and `counterpart_idx` columns.

To prepare data for an offline machine, set `HUB_CACHE` to a writable cache directory and run the downloader on a connected machine.

```bash
export HUB_CACHE
python3 - <<'PY'
import os
import runpy
from huggingface_hub import snapshot_download

datasets = runpy.run_path("common/epd_replay.py")["QUALITY_DATASETS"]
for slug, revision, _ in datasets.values():
    snapshot_download(
        repo_id=slug.replace("--", "/", 1),
        repo_type="dataset",
        revision=revision,
        cache_dir=os.environ["HUB_CACHE"],
        allow_patterns=["data/*.parquet"],
    )
PY
```

Transfer the three `datasets--*` directories from the cache, preserving each directory's `blobs/` and `snapshots/` layout and relative symlinks. Use `tar` with its default symlink behavior or `cp -a`. Set `HUB_CACHE` on the receiving machine to the directory containing the three transferred directories.

## Offline quality scoring

Set `LMMS_EVAL` to the pinned checkout, `HUB_CACHE` to the prepared Hub cache, and `HF_DATASETS_CACHE` to a writable directory for derived Arrow data. The task prompt and score come from the pinned evaluator. Natural stopping remains enabled.

```bash
export PYTHONPATH="$LMMS_EVAL${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_DATASETS_CACHE
task=ocrbench

python3 common/epd_replay.py generate-quality \
  --task "$task" --lmms-eval "$LMMS_EVAL" --hub-cache "$HUB_CACHE" \
  --trace "$RUN_DIR/$task.trace.jsonl" --qps 1 --seed 42

python3 common/epd_replay.py replay \
  --model "$MODEL_ID" --trace "$RUN_DIR/$task.trace.jsonl" \
  --group B --endpoints "$PD_ENDPOINT" --output "$RUN_DIR/$task.B.jsonl"

python3 common/epd_replay.py score-quality \
  --task "$task" --lmms-eval "$LMMS_EVAL" --hub-cache "$HUB_CACHE" \
  --trace "$RUN_DIR/$task.trace.jsonl" --responses "$RUN_DIR/$task.B.jsonl" \
  --output "$RUN_DIR/$task.B.score.json"
```

Use the same trace for every group. The scorer reads the captured response log and writes a new score file. It reports a full-set official score when every test row has a completed response. Failed or missing requests remain visible in the submitted denominator.

Compare two score reports on that frozen set. The paired bootstrap samples images for OCRBench, questions for RealWorldQA, and reciprocal pairs for MuirBench. It reports the change in percentage points with a 95% interval, score transitions, and both groups' completion counts. The submitted-score comparison counts service failures as uncredited. Use it as the task accuracy comparison only when both reports have full official coverage.

```bash
python3 common/compare_quality.py --base "$B_SCORES" --test "$C70_SCORES" \
  --ocr-responses "$B_OCR_RESPONSES" "$C70_OCR_RESPONSES" \
  --output "$RUN_DIR/ocrbench.B-to-C70.paired.json"
```

For RealWorldQA or MuirBench, omit `--ocr-responses`. MuirBench also reports the change in the fraction of pairs answered correctly on both questions.

For the MuirBench load pilot, select one reciprocal question pair per task and maximum image count within the pair. Record each pair's two `doc_id`, `idx`, `counterpart_idx`, `task` and `images` values in the selection JSON, with the dataset `revision`, total `rows`, `arrival_seed` and `candidate_qps`. The experiment's selection covers 36 strata and 72 questions. Extract the selected rows from the frozen trace and use the same pilot across configurations.

```bash
python3 common/muir_pilot_trace.py --frozen-trace "$MUIR_FULL_TRACE" \
  --selection "$MUIR_PILOT_SELECTION" --output "$MUIR_PILOT_TRACE"
```

At a lower candidate QPS, pass `--qps` and a new output path. The extraction keeps the chosen question order, image bytes and prompts while regenerating only planned arrivals. A qualified pilot completes all 72 requests.

For the full MuirBench run, use the same original trace in all four groups. Pass `replay --arrival-scale` with the original trace QPS divided by the qualified shared QPS. For example, a trace generated at 2 QPS needs `--arrival-scale 2` to run at 1 QPS. This changes the planned offsets without creating another copy of the full image trace. Record the scale with the run and preserve failed attempts before lowering it.

Replay the same 72-question pilot with `replay --serial` and an unused output path to check whether concurrent execution changes greedy answers. Serial mode waits for the previous request to finish before sending the next and keeps the frozen request order, endpoints and generation settings. Save both response logs and match by question ID and ordered image hashes.

## 7B raw evidence audit

```bash
python3 common/audit_7b.py --run-dir "$RUN_DIR" --b-label no-probe \
  --muir-label 1qps --output "$RUN_DIR/7b-raw-evidence-audit.json"
```

The audit checks every planned matrix cell and its pairing, long-generation interference repeats, quality and author-reference outputs, and the C100/C70/C50 feature-layout probes. From the receiver logs it verifies that compact prompts preserve the measured first decode position. It records verified, missing and invalid checks, including failures inside complete request logs. Use the label of the qualified full MuirBench replay. Before publishing the 7B freeze, review the audit alongside the traffic and stage-timing analyses, rendered figures and service releases.

When the measurements are complete, build the analysis bundle with Matplotlib in the Python environment.

```bash
python3 common/build_7b_report.py --run-dir "$RUN_DIR" --output-dir "$RUN_DIR/analysis-7b" \
  --client-host "$PD_CLIENT_IP" --b-label no-probe --muir-label 1qps
```

The bundle uses a new output directory, runs the raw audit first, and saves figures, paired quality comparisons, traffic and stage statistics. `analysis-receipt.json` records hashes after all commands succeed. Review those outputs and the release records before publishing the separate 7B freeze.
