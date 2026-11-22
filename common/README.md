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
