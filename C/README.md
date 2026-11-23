# C · SGLang E/PD with VisionZip

C uses the same Qwen2.5-VL-7B-Instruct checkpoint and E/PD placement as B. The encoder uses the final ViT block's post-RoPE query and key tensors to score tokens. It selects dominant tokens, merges the remaining features into contextual representatives, and sends only the retained visual feature rows. PD uses the retained indices to shrink the prompt and preserve original M-RoPE coordinates. The patch preserves the vision attention backend.

The patch publishes each compressed image's retained row count before the forward pass and preserves request retirement when the encoder fails before sending a token layout. CPU checks cover both paths. The revised runtime passed six two-node C100 identity probes and six C70 payload probes. The C70 probes matched each image's sent rows, received rows, and visual placeholders in serial and mixed workloads.

## Apply and check

Use a separate checkout so that B can continue to use the original SGLang source. Use the same base version and runtime dependencies on both nodes. The CPU test can run with the standalone PyTorch CPU wheel and does not require SGLang imports, model weights, or a dataset.

```bash
# From the root of this reproduction repository.
REPRO="$PWD"
BASE=8396d60a14ef03b23d2cdaa5c64adb18eba7c703
git clone https://github.com/sgl-project/sglang.git ../sglang-c
git -C ../sglang-c checkout --detach "$BASE"
git -C ../sglang-c apply --check "$REPRO/C/visionzip-epd-qwen25.patch"
git -C ../sglang-c apply "$REPRO/C/visionzip-epd-qwen25.patch"
python C/visionzip_cpu_checks.py ../sglang-c
python common/diagnostics/epd_unadmitted_cpu_checks.py ../sglang-c
```

The checks cover the author-style score and merge formulas, C100 identity, malformed indices, multi-image prompt and decode positions, out-of-order embedding parts, compressed BF16 metadata, and retirement before token-layout assembly. For serving, install the fixed base build's dependencies and launch from the patched sources on both nodes. [SGLang installation and base E/PD setup](https://github.com/sgl-project/sglang/blob/8396d60a14ef03b23d2cdaa5c64adb18eba7c703/docs/docs/advanced_features/epd_disaggregation.mdx)

Run `python C/visionzip_cpu_checks.py ../sglang-c "$SGLANG_B_ROOT"` to use a separate pinned native B checkout for the comparison. The test uses the Qwen2.5-VL position builder and the scheduler's decode position expansion for 1, 4 and 8 images, including a non-square 1080p-derived grid.

## Run C100, C70, and C50

Set `SGLANG_ROOT` to the local patched checkout, `MODEL_PATH` to a local copy of the same pinned checkpoint, and `ENCODER_IP` and `PD_IP` to your two experiment addresses. Set `GPU_UUID` on each node to its allocated GPU. Use the same pixel budget as B. Leave the optional prefix and global multimodal caches off and disable the PD radix cache for the comparison.

```bash
# Encoder node. Start with 100, then restart at 70 and 50 after validation.
RATIO=100
CUDA_VISIBLE_DEVICES="$GPU_UUID" \
PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}" \
SGLANG_HOST_IP="$ENCODER_IP" SGLANG_VIT_ENABLE_CUDA_GRAPH=0 \
SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU=1 \
SGLANG_EPD_VISIONZIP_RATIO="$RATIO" \
python -m sglang.launch_server --model-path "$MODEL_PATH" \
  --served-model-name "${MODEL_ID:-qwen2.5-vl-7b}" --dtype bfloat16 --tp-size 1 \
  --attention-backend fa3 --mm-attention-backend fa3 \
  --host "$ENCODER_IP" --port 30000 --encoder-only \
  --encoder-transfer-backend zmq_to_scheduler
```

```bash
# PD node. Send benchmark requests to this service or to the same router used for B.
CUDA_VISIBLE_DEVICES="$GPU_UUID" \
PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}" \
SGLANG_HOST_IP="$PD_IP" SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_INTERVAL=0 \
python -m sglang.launch_server --model-path "$MODEL_PATH" \
  --served-model-name "${MODEL_ID:-qwen2.5-vl-7b}" --dtype bfloat16 --tp-size 1 \
  --attention-backend fa3 --mm-attention-backend fa3 \
  --mem-fraction-static 0.65 --context-length 128000 --chunked-prefill-size 8192 \
  --host "$PD_IP" --port 30002 --language-only \
  --encoder-urls "http://$ENCODER_IP:30000" \
  --encoder-transfer-backend zmq_to_scheduler --disable-radix-cache --enable-metrics
```

| Configuration | Token handling | Acceptance check |
|---|---|---|
| C100 | Keep all visual rows and carry identity indices through the new protocol. | Compare native B and C100 on the same single- and multi-image requests. Match visual row counts and tensor bytes, then check greedy token IDs, prompt layout, and first-token log probabilities. C100 adds index metadata. |
| C70 | Keep about 65% dominant rows and 5% contextual rows per image. | Verify the compact image segments and record exact retained rows per image, including rounding. |
| C50 | Keep about 45% dominant rows and 5% contextual rows per image. | Repeat the same checks and compare actual bytes, quality, and performance against B. |

`SGLANG_EPD_VISIONZIP_RATIO=0` selects the standard encoder path. The adaptation targets image-only Qwen2.5-VL, a TP1 encoder without vision data-parallel sharding, and HTTP with `zmq_to_scheduler`. Compressed runs require ViT CUDA graphs to be off. Use the same graph setting in B and C. For diagnostics, export `SGLANG_EPD_VISIONZIP_TRACE=1` on both nodes before starting C. The encoder logs feature and metadata frame bytes and per-image row counts. PD logs received rows, visual placeholders, and M-RoPE sequence length. Frame bytes exclude TCP and IP overhead. For formal timing, leave the trace off.

Capture six untimed identity requests before stopping B. Pass three frozen traces whose first rows contain one, four, and eight images, respectively. Repeat against C100 using the same traces. The script checks the scheduler's final prompt length and the 16 generated token IDs per request, then reports first-token log-probability differences for serial and mixed concurrent requests. Input IDs returned by the tokenizer can precede E/PD prompt assembly. With the transfer logs the check also verifies that the sent rows, received rows, and placeholders agree for each image. The CPU checks verify exact M-RoPE coordinates.

```bash
python3 C/check_identity.py --endpoint "$B_PD_ENDPOINT" --model "$MODEL_ID" \
  --traces "$ONE_IMAGE_TRACE" "$FOUR_IMAGE_TRACE" "$EIGHT_IMAGE_TRACE" \
  --output "$RUN_DIR/identity.B.jsonl"
python3 C/check_identity.py --endpoint "$C100_PD_ENDPOINT" --model "$MODEL_ID" \
  --traces "$ONE_IMAGE_TRACE" "$FOUR_IMAGE_TRACE" "$EIGHT_IMAGE_TRACE" \
  --output "$RUN_DIR/identity.C100.jsonl" --reference "$RUN_DIR/identity.B.jsonl" \
  --transfer-logs "$C100_ENCODER_LOG" "$C100_PD_LOG"
```

Before each timed compressed run, replay the same untimed probes with `--ratio 70` or `--ratio 50`. Use the C100 response log as `--reference`, its encoder log as `--reference-encoder-log`, and the compressed run's two logs as `--transfer-logs`. The verifier checks actual feature bytes, exact per-image retention after rounding, and the corresponding prompt-length change.

## Author reference, single GPU

`visionzip_reference.py` uses standard Transformers at 100% and the authors' implementation at 70% and 50%. It requires a CUDA PyTorch environment with one visible GPU, `transformers==4.49.0`, `qwen-vl-utils==0.0.8`, Pillow, and Accelerate. The default eager attention backend avoids a FlashAttention dependency. Install a compatible FlashAttention 2 build before setting `--attention flash_attention_2`.

The compressed reference uses a local cache-layout compatibility fix. The pinned author code shrinks the visual sequence while carrying the original `cache_position` into the eager decoder. The runner applies the fix to its temporary source copy, preserving the author's token selection, merges, and M-RoPE coordinates. The decoder receives contiguous DynamicCache indices and the matching attention mask at prefill and every decode step. Record the run as the author implementation with this compatibility fix.

```bash
git clone https://github.com/JIA-Lab-research/VisionZip.git ../visionzip-author
git -C ../visionzip-author checkout --detach 8f86b55c6f000eb033e6912538af2dd7dcb30502
python C/visionzip_reference_cpu_checks.py ../visionzip-author/Qwen2_5_VL/qwen2_5vl_visionzip.py
python C/visionzip_reference.py --model "$MODEL_PATH" --ratio 70 \
  --author-source ../visionzip-author/Qwen2_5_VL/qwen2_5vl_visionzip.py \
  --trace "$QUALITY_TRACE" --output "$REFERENCE_OUTPUT"
```

Generate the trace with `common/epd_replay.py generate-quality` and choose `ocrbench` or `realworldqa`. The JSONL trace has one item per line, with one PNG data URI and a SHA-256 for the decoded bytes. Freeze the prompts and bytes before comparing. Score the runner's answer rows with the same trace and task scorer. [Shared replay and scoring](../common/README.md)

## Upstream attribution

`LICENSE.sglang` is the unmodified Apache 2.0 license from the pinned SGLang commit. The patch contains local changes to SGLang's encoder, receiver, model, and multimodal processor. [Upstream SGLang license](https://github.com/sgl-project/sglang/blob/8396d60a14ef03b23d2cdaa5c64adb18eba7c703/LICENSE)

The runner loads the authors' VisionZip source from a separate checkout and applies the ratio and cache-layout changes to a temporary copy. The source header credits the Qwen Team, Hugging Face, and Senqiao Yang. [VisionZip license](https://github.com/JIA-Lab-research/VisionZip/blob/8f86b55c6f000eb033e6912538af2dd7dcb30502/LICENSE)
