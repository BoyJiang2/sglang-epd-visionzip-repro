#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash A/start.sh [--dry-run]
Required: SGLANG_ROOT, MODEL_PATH, GPU_UUID
Optional: HOST (127.0.0.1), PORT (30002), MODEL_ID (qwen2.5-vl-7b), PYTHON_BIN (python3)
Start one complete 7B replica. Launch separately on each of the two hosts.
HELP
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
if (( $# > 1 )) || [[ -n "${1:-}" && "$1" != --dry-run ]]; then usage >&2; exit 2; fi

revision=8396d60a14ef03b23d2cdaa5c64adb18eba7c703
: "${SGLANG_ROOT:?Set the fixed SGLang checkout root}"
: "${MODEL_PATH:?Set the local 7B model path}"
: "${GPU_UUID:?Set the allocated GPU UUID}"
host=${HOST:-127.0.0.1}
port=${PORT:-30002}
[[ "$port" =~ ^[1-9][0-9]{0,4}$ ]] && (( port <= 65535 )) || { echo "Invalid PORT" >&2; exit 2; }
[[ "$GPU_UUID" =~ ^GPU-[[:xdigit:]]{8}(-[[:xdigit:]]{4}){3}-[[:xdigit:]]{12}$ ]] || {
  echo "GPU_UUID must contain one explicit GPU UUID" >&2
  exit 2
}

cmd=("${PYTHON_BIN:-python3}" -m sglang.launch_server
     --model-path "$MODEL_PATH" --served-model-name "${MODEL_ID:-qwen2.5-vl-7b}"
     --host "$host" --port "$port" --dtype bfloat16 --tp-size 1
     --mem-fraction-static 0.65 --context-length 128000 --chunked-prefill-size 8192
     --attention-backend fa3 --mm-attention-backend fa3 --sampling-backend flashinfer
     --disable-radix-cache --enable-metrics)
if [[ "${1:-}" == --dry-run ]]; then
  printf 'SGLang revision %s\nCUDA_VISIBLE_DEVICES=%q\n' "$revision" "$GPU_UUID"
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
