#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash B/start.sh encoder|pd [--dry-run]
Required: SGLANG_ROOT, MODEL_PATH, GPU_UUID
PD role also requires ENCODER_URL, the encoder's reachable HTTP(S) URL.
Optional: HOST (127.0.0.1), SGLANG_HOST_IP (HOST), PORT (30000 encoder / 30002 PD),
          MODEL_ID (qwen2.5-vl-7b), PYTHON_BIN (python3)
HELP
}
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  encoder|pd) role=$1 ;;
  *) usage >&2; exit 2 ;;
esac
if (( $# > 2 )) || [[ -n "${2:-}" && "$2" != --dry-run ]]; then usage >&2; exit 2; fi

revision=8396d60a14ef03b23d2cdaa5c64adb18eba7c703
: "${SGLANG_ROOT:?Set the fixed SGLang checkout root}"
: "${MODEL_PATH:?Set the local 7B model path}"
: "${GPU_UUID:?Set the allocated GPU UUID}"
host=${HOST:-127.0.0.1}
[[ "$GPU_UUID" =~ ^GPU-[[:xdigit:]]{8}(-[[:xdigit:]]{4}){3}-[[:xdigit:]]{12}$ ]] || {
  echo "GPU_UUID must contain one explicit GPU UUID" >&2
  exit 2
}
cmd=("${PYTHON_BIN:-python3}" -m sglang.launch_server
     --model-path "$MODEL_PATH" --served-model-name "${MODEL_ID:-qwen2.5-vl-7b}"
     --host "$host" --dtype bfloat16 --tp-size 1 --mem-fraction-static 0.65
     --attention-backend fa3 --mm-attention-backend fa3 --sampling-backend flashinfer
     --encoder-transfer-backend zmq_to_scheduler)
export SGLANG_HOST_IP="${SGLANG_HOST_IP:-$host}"
if [[ "$role" == encoder ]]; then
  port=${PORT:-30000}
  export SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU=1
  cmd+=(--encoder-only)
else
  port=${PORT:-30002}
  [[ "${ENCODER_URL:-}" =~ ^https?://[^[:space:]]+$ ]] || {
    echo "Set ENCODER_URL to a reachable HTTP(S) URL" >&2
    exit 2
  }
  cmd+=(--language-only --encoder-urls "$ENCODER_URL"
        --context-length 128000 --chunked-prefill-size 8192
        --disable-radix-cache --enable-metrics)
fi
[[ "$port" =~ ^[1-9][0-9]{0,4}$ ]] && (( port <= 65535 )) || { echo "Invalid PORT" >&2; exit 2; }
cmd+=(--port "$port")
if [[ "${2:-}" == --dry-run ]]; then
  printf 'SGLang revision %s\nCUDA_VISIBLE_DEVICES=%q\nSGLANG_HOST_IP=%q\n' \
    "$revision" "$GPU_UUID" "$SGLANG_HOST_IP"
  if [[ "$role" == encoder ]]; then printf 'SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU=1\n'; fi
  printf '%q ' "${cmd[@]}"; printf '\n'
  exit 0
fi

[[ "$(git -C "$SGLANG_ROOT" rev-parse HEAD)" == "$revision" ]] &&
  git -C "$SGLANG_ROOT" diff --quiet HEAD -- python || {
  echo "Use the unmodified Python source at SGLang $revision" >&2
  exit 2
}
[[ -r "$MODEL_PATH/config.json" ]] || { echo "Cannot read model config" >&2; exit 2; }
export PYTHONPATH="$SGLANG_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
exec "${cmd[@]}"
