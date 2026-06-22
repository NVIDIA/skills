#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# jetson-llm-benchmark: run `vllm bench serve` against a running vLLM server,
# parse output into the skill's JSON contract.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
. "${SKILLS_ROOT}/jetson-diagnostic/scripts/detect_jetson.sh"

usage() {
  cat <<'EOF'
Usage: bench_vllm.sh --model <id> [options]
  --model <id>           Model id served by the running vLLM server
  --endpoint <url>       Default: http://localhost:8000
  --concurrency <list>   Comma-separated, default: 1,8
  --input-len <n>        Default: 2048
  --output-len <n>       Default: 128
  --num-prompts <n>      Default: 50
  --no-warmup            Skip warmup pass
  --container <image>    Override benchmark client container
  --native               Run host-native vllm bench serve instead of a container
EOF
}

MODEL=""
ENDPOINT="http://localhost:8000"
CONC="1,8"
ILEN=2048
OLEN=128
NP=50
WARMUP=1
CONTAINER=""
BENCH_MODE="container"
FORCE_NATIVE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --endpoint) ENDPOINT="$2"; shift 2;;
    --concurrency) CONC="$2"; shift 2;;
    --input-len) ILEN="$2"; shift 2;;
    --output-len) OLEN="$2"; shift 2;;
    --num-prompts) NP="$2"; shift 2;;
    --no-warmup) WARMUP=0; shift;;
    --container) CONTAINER="$2"; shift 2;;
    --native) FORCE_NATIVE=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

[[ -z "${MODEL}" ]] && { echo "ERROR: --model is required" >&2; usage; exit 2; }

l4t_major() {
  local raw="${JETSON_L4T_VERSION:-}"
  raw="${raw#r}"
  raw="${raw#R}"
  local major="${raw%%.*}"
  [[ "${major}" =~ ^[0-9]+$ ]] && printf '%s\n' "${major}" || printf '0\n'
}

orin_uses_upstream_vllm() {
  [[ "${JETSON_GENERATION:-}" = "orin" && "$(l4t_major)" -ge 39 ]]
}

if [[ "${FORCE_NATIVE}" -eq 1 ]]; then
  BENCH_MODE="native"
  CONTAINER=""
elif [[ -z "${CONTAINER}" ]]; then
  case "${JETSON_GENERATION:-}" in
    thor) CONTAINER="vllm/vllm-openai:latest"; BENCH_MODE="container";;  # Thor requires upstream vLLM 0.20+.
    orin)
      if orin_uses_upstream_vllm; then
        CONTAINER="vllm/vllm-openai:latest"
      else
        CONTAINER="ghcr.io/nvidia-ai-iot/vllm:latest-jetson-orin"
      fi
      BENCH_MODE="container"
      ;;
    *) echo "ERROR: cannot pick benchmark mode for generation '${JETSON_GENERATION:-unknown}' (sku '${JETSON_SKU:-unknown}')" >&2; exit 3;;
  esac
else
  BENCH_MODE="container"
fi

WARN=()
if command -v nvpmodel >/dev/null 2>&1; then
  NVPMODEL_OUT="$(nvpmodel -q 2>/dev/null || true)"
  NVPMODEL_NAME="$(printf '%s\n' "${NVPMODEL_OUT}" | awk -F': ' '/NV Power Mode/ {print $2; exit}')"
  NVPMODEL_ID="$(printf '%s\n' "${NVPMODEL_OUT}" | awk '/NV Power Mode/ {getline; print $1; exit}')"
  case "${NVPMODEL_NAME}" in
    MAXN|MAXN_*) ;;
    *) WARN+=("nvpmodel is ${NVPMODEL_NAME:-unknown} (id ${NVPMODEL_ID:-unknown}); MAXN/MAXN_* preferred for comparable results") ;;
  esac
fi

run_bench_cmd() {
  if [[ "${BENCH_MODE}" = "container" ]]; then
    docker run --rm --network host "${CONTAINER}" vllm bench serve "$@"
  else
    vllm bench serve "$@"
  fi
}

run_one() {
  local c="$1" tmp
  tmp="$(mktemp)"
  local extra_warm=()
  [[ "${WARMUP}" -eq 1 ]] && extra_warm=(--num-prompts 10 --max-concurrency "${c}")

  if [[ "${#extra_warm[@]}" -gt 0 ]]; then
    run_bench_cmd \
      --backend vllm \
      --base-url "${ENDPOINT}" \
      --model "${MODEL}" \
      --dataset-name random \
      --random-input-len "${ILEN}" --random-output-len "${OLEN}" \
      "${extra_warm[@]}" >/dev/null 2>&1 || true
  fi

  run_bench_cmd \
    --backend vllm \
    --base-url "${ENDPOINT}" \
    --model "${MODEL}" \
    --dataset-name random \
    --random-input-len "${ILEN}" --random-output-len "${OLEN}" \
    --num-prompts "${NP}" --max-concurrency "${c}" \
    > "${tmp}" 2>&1

  python3 - "${c}" "${tmp}" <<'PY'
import json, re, sys
c = int(sys.argv[1])
with open(sys.argv[2]) as f:
    txt = f.read()

def grab(pat):
    m = re.search(pat, txt)
    return float(m.group(1)) if m else 0.0

print(json.dumps({
    "concurrency": c,
    "ttft_ms_p50":  grab(r"Median TTFT \(ms\):\s*([\d.]+)"),
    "ttft_ms_p99":  grab(r"P99 TTFT \(ms\):\s*([\d.]+)"),
    "itl_ms_p50":   grab(r"Median ITL \(ms\):\s*([\d.]+)"),
    "itl_ms_p99":   grab(r"P99 ITL \(ms\):\s*([\d.]+)"),
    "tpot_ms_p50":  grab(r"Median TPOT \(ms\):\s*([\d.]+)"),
    "throughput_tok_s": grab(r"Output token throughput \(tok/s\):\s*([\d.]+)"),
    "e2e_latency_ms_p50": grab(r"Median E2EL \(ms\):\s*([\d.]+)"),
}))
PY
  rm -f "${tmp}"
}

RESULTS="["
FIRST=1
IFS=',' read -ra CONC_ARR <<< "${CONC}"
for c in "${CONC_ARR[@]}"; do
  [[ "${FIRST}" -eq 1 ]] || RESULTS+=","
  FIRST=0
  RESULTS+="$(run_one "${c}")"
done
RESULTS+="]"

WARN_JSON="[]"
if [[ "${#WARN[@]}" -gt 0 ]]; then
  WARN_JSON="$(printf '%s\n' "${WARN[@]}" | python3 -c 'import sys,json;print(json.dumps([l.rstrip() for l in sys.stdin if l.strip()]))')"
fi

BENCH_MODEL="${MODEL}" \
BENCH_CONTAINER="${CONTAINER}" \
BENCH_MODE="${BENCH_MODE}" \
BENCH_RESULTS="${RESULTS}" \
BENCH_WARN="${WARN_JSON}" \
BENCH_ILEN="${ILEN}" \
BENCH_OLEN="${OLEN}" \
BENCH_NP="${NP}" \
JETSON_SKU="${JETSON_SKU:-unknown}" \
JETSON_GENERATION="${JETSON_GENERATION:-unknown}" \
JETSON_PRODUCT_LINE="${JETSON_PRODUCT_LINE:-unknown}" \
JETSON_VARIANT="${JETSON_VARIANT:-unknown}" \
JETSON_L4T_VERSION="${JETSON_L4T_VERSION:-unknown}" \
python3 - <<'PY'
import json, os
runs = json.loads(os.environ["BENCH_RESULTS"])
out = {
  "skill": "jetson-llm-benchmark",
  "runtime": "vllm",
  "model": os.environ["BENCH_MODEL"],
  "sku": os.environ["JETSON_SKU"],
  "generation": os.environ["JETSON_GENERATION"],
  "product_line": os.environ["JETSON_PRODUCT_LINE"],
  "variant": os.environ["JETSON_VARIANT"],
  "l4t": os.environ["JETSON_L4T_VERSION"],
  "benchmark_mode": os.environ["BENCH_MODE"],
  "container": os.environ["BENCH_CONTAINER"],
  "config": {
    "input_len":   int(os.environ["BENCH_ILEN"]),
    "output_len":  int(os.environ["BENCH_OLEN"]),
    "num_prompts": int(os.environ["BENCH_NP"]),
  },
  "runs": runs,
  "warnings": json.loads(os.environ["BENCH_WARN"]),
}
print(json.dumps(out, indent=2))
PY
