#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == --help ]]; then
  echo 'Set MODEL_PATH, RUN_DIR, A_ENDPOINT_1 and A_ENDPOINT_2. Optionally set MODEL_ID and PYTHON_BIN.'
  exit 0
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${PYTHON_BIN:-python3}"
model="${MODEL_ID:-qwen2.5-vl-7b}"
model_path="${MODEL_PATH:?Set the local checkpoint path for the data generator}"
run="${RUN_DIR:?Set the run directory}"
endpoints=("${A_ENDPOINT_1:?Set the first service origin}" "${A_ENDPOINT_2:?Set the second service origin}")
bench=("$python" "$root/common/epd_replay.py")
mkdir -p "$run"
plan="$run/formal-design.json"
if [[ ! -e "$plan" ]]; then
  (set -o noclobber; cat "$root/plans/formal-design.json" > "$plan")
fi

schedule="$("$python" - "$plan" <<'PY'
import json
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
timeout = plan["timeout_per_request_s"]
for seed in plan["seeds"]:
    for images, rates in plan["rates_qps"].items():
        for rate in rates:
            print("formal", images, rate, seed, plan["requests_per_cell"], plan["output_tokens"], timeout)
official = plan["paper"]
for rate in official["target_rates"]:
    print("official", official["random_images_upper_bound"], rate, official["seed"],
          official["requests"], official["output_tokens"], timeout)
PY
)"

while read -r kind images rate seed requests tokens timeout; do
  case_id="$kind-${images}img-${rate}qps-seed$seed"
  trace="$run/$case_id.trace.jsonl"
  responses="$run/$case_id.A.jsonl"
  if [[ -e "$responses" ]]; then
    [[ -f "$trace" && "$(wc -l < "$responses")" -eq "$requests" ]] || {
      echo "$case_id has a partial original result. Review it before resuming." >&2
      exit 1
    }
    echo "FORMAL_CELL_RETAINED $case_id"
    continue
  fi
  if [[ ! -e "$trace" ]]; then
    additional=()
    if [[ "$kind" == official ]]; then additional+=(--random-images); fi
    "${bench[@]}" generate --trace "$trace" --model "$model_path" --qps "$rate" \
      --images "$images" --seed "$seed" --requests "$requests" --output-tokens "$tokens" "${additional[@]}"
  fi
  [[ "$(wc -l < "$trace")" -eq "$requests" ]] || {
    echo "$case_id trace has an unexpected number of requests." >&2
    exit 1
  }
  echo "FORMAL_CELL_BEGIN $case_id"
  status=0
  "${bench[@]}" replay --trace "$trace" --output "$responses" --model "$model" --group A \
    --endpoints "${endpoints[@]}" --timeout "$timeout" || status=$?
  echo "FORMAL_CELL_END $case_id status=$status"
  if (( status != 0 )); then
    for endpoint in "${endpoints[@]}"; do curl -fsS --max-time 4 "${endpoint%/}/health" >/dev/null; done
  fi
done <<< "$schedule"
echo FORMAL_A_SYNTHETIC_COMPLETE
