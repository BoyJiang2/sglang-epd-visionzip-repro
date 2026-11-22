#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: bash 235B/start.sh colocated|encoder|pd [--dry-run]
Required: SGLANG_ROOT, MODEL_PATH, GPU_UUIDS
GPU_UUIDS contains exactly one encoder UUID or four comma-separated TP4 UUIDs.
PD also requires one or two space-separated URLs in ENCODER_URLS.
Optional: HOST (127.0.0.1), PORT (31231 encoder / 31230 colocated or PD),
          MODEL_ID (qwen3-vl-235b-a22b-instruct-fp8), PYTHON_BIN (python3)
HELP
}
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
  colocated|encoder|pd) role=$1 ;;
  *) usage >&2; exit 2 ;;
esac
if (( $# > 2 )) || [[ -n "${2:-}" && "$2" != --dry-run ]]; then usage >&2; exit 2; fi

revision=8396d60a14ef03b23d2cdaa5c64adb18eba7c703
: "${SGLANG_ROOT:?Set the fixed SGLang checkout root}"
: "${MODEL_PATH:?Set the local 235B FP8 model path}"
: "${GPU_UUIDS:?Set the allocated GPU UUIDs}"
host=${HOST:-127.0.0.1}
uuid='GPU-[[:xdigit:]]{8}(-[[:xdigit:]]{4}){3}-[[:xdigit:]]{12}'
[[ "$GPU_UUIDS" =~ ^$uuid(,$uuid)*$ ]] || { echo "Invalid GPU_UUIDS" >&2; exit 2; }
IFS=, read -r -a devices <<< "$GPU_UUIDS"
declare -A seen=()
for device in "${devices[@]}"; do
  [[ ! -v "seen[$device]" ]] || { echo "GPU UUIDs must be distinct" >&2; exit 2; }
  seen[$device]=1
done

cmd=("${PYTHON_BIN:-python3}" -m sglang.launch_server
     --model-path "$MODEL_PATH" --served-model-name "${MODEL_ID:-qwen3-vl-235b-a22b-instruct-fp8}"
     --host "$host" --dtype bfloat16)
export SGLANG_HOST_IP="${SGLANG_HOST_IP:-$host}"
if [[ "$role" == encoder ]]; then
  (( ${#devices[@]} == 1 )) || { echo "Encoder requires one GPU UUID" >&2; exit 2; }
  port=${PORT:-31231}
  export SGLANG_ENCODER_IMAGE_PROCESSOR_USE_GPU=1
  cmd+=(--tp-size 1 --encoder-only --encoder-transfer-backend zmq_to_scheduler)
else
  (( ${#devices[@]} == 4 )) || { echo "Colocated or PD requires four GPU UUIDs" >&2; exit 2; }
  port=${PORT:-31230}
  cmd+=(--tp-size 4 --nnodes 1 --ep-size 1 --disable-radix-cache --enable-metrics
        --context-length 32768 --mem-fraction-static 0.85
        --fp8-gemm-backend auto --moe-runner-backend auto)
  if [[ "$role" == pd ]]; then
    read -r -a urls <<< "${ENCODER_URLS:?Set one or two reachable encoder URLs}"
    (( ${#urls[@]} >= 1 && ${#urls[@]} <= 2 )) || { echo "PD requires one or two encoder URLs" >&2; exit 2; }
    for url in "${urls[@]}"; do
      [[ "$url" =~ ^https?://[^[:space:]]+$ ]] || { echo "Invalid encoder URL" >&2; exit 2; }
    done
    if (( ${#urls[@]} == 2 )) && [[ "${urls[0]}" == "${urls[1]}" ]]; then
      echo "Encoder URLs must be distinct" >&2
      exit 2
    fi
    cmd+=(--language-only --encoder-transfer-backend zmq_to_scheduler --encoder-urls "${urls[@]}")
  fi
fi
[[ "$port" =~ ^[1-9][0-9]{0,4}$ ]] && (( port <= 65535 )) || { echo "Invalid PORT" >&2; exit 2; }
cmd+=(--port "$port")
if [[ "${2:-}" == --dry-run ]]; then
  printf 'SGLang revision %s\nCUDA_VISIBLE_DEVICES=%q\nSGLANG_HOST_IP=%q\n' \
    "$revision" "$GPU_UUIDS" "$SGLANG_HOST_IP"
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
export CUDA_VISIBLE_DEVICES="$GPU_UUIDS"
exec "${cmd[@]}"
