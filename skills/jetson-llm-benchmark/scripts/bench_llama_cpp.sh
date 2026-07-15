#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# jetson-llm-benchmark: run llama.cpp's `llama-bench` in the
# NVIDIA-AI-IOT prebuilt llama_cpp container, parse output to JSON.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLS_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
. "${SKILLS_ROOT}/jetson-diagnostic/scripts/detect_jetson.sh"

usage() {
  cat <<'EOF'
Usage: bench_llama_cpp.sh --model <path-to-gguf> [options]
  --model <path>         Path on host to a .gguf file
  --n-prompt <n>         Prompt-eval tokens, default: 512
  --n-gen <n>            Generation tokens, default: 128
  --n-gpu-layers <n>     Layers offloaded to GPU, default: 99 (all)
  --threads <n>          CPU threads, default: auto
  --container <image>    Override auto-selected ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-{orin,thor}
EOF
}

MODEL=""
NP=512
NG=128
NGL=99
THREADS=""
case "${JETSON_GENERATION:-unknown}" in
  orin) CONTAINER="ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin";;
  thor) CONTAINER="ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-thor";;
  *) CONTAINER="ghcr.io/nvidia-ai-iot/llama_cpp:latest";;
esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --n-prompt) NP="$2"; shift 2;;
    --n-gen) NG="$2"; shift 2;;
    --n-gpu-layers) NGL="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;
    --container) CONTAINER="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

[[ -z "${MODEL}" ]] && { echo "ERROR: --model is required" >&2; usage; exit 2; }
[[ -f "${MODEL}" ]] || { echo "ERROR: model file not found: ${MODEL}" >&2; exit 2; }

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

MODEL_DIR="$(cd "$(dirname "${MODEL}")" && pwd)"
MODEL_FILE="$(basename "${MODEL}")"

THREAD_ARGS=()
[[ -n "${THREADS}" ]] && THREAD_ARGS=(-t "${THREADS}")

TMP="$(mktemp)"
TMP_ERR="$(mktemp)"
trap 'rm -f "${TMP}" "${TMP_ERR}"' EXIT
docker run --rm --runtime nvidia \
  -v "${MODEL_DIR}:/models:ro" \
  "${CONTAINER}" \
  llama-bench \
    -m "/models/${MODEL_FILE}" \
    -ngl "${NGL}" \
    -p "${NP}" -n "${NG}" \
    -o csv \
    "${THREAD_ARGS[@]}" \
  > "${TMP}" 2> "${TMP_ERR}"

# Surface llama-bench stderr as a warning rather than mixing it into the CSV.
if [[ -s "${TMP_ERR}" ]]; then
  WARN+=("llama-bench stderr: $(head -n1 "${TMP_ERR}")")
fi

WARN_JSON="[]"
if [[ "${#WARN[@]}" -gt 0 ]]; then
  WARN_JSON="$(printf '%s\n' "${WARN[@]}" | python3 -c 'import sys,json;print(json.dumps([l.rstrip() for l in sys.stdin if l.strip()]))')"
fi

python3 - "${TMP}" <<PY
import csv, json, sys

with open(sys.argv[1]) as f:
    lines = [l for l in f if l.strip() and not l.startswith("#")]
reader = csv.DictReader(lines)
rows = list(reader)

NP = ${NP}
NG = ${NG}

def as_float(row, *keys):
    for key in keys:
        try:
            return float(row.get(key, 0) or 0)
        except ValueError:
            continue
    return 0.0

def as_int(row, key):
    try:
        return int(float(row.get(key, 0) or 0))
    except ValueError:
        return 0

def find_legacy(rows, key, op):
    for r in rows:
        if r.get("test","") == op:
            return as_float(r, key)
    return 0.0

def find_current(rows, prompt_tokens, gen_tokens):
    for r in rows:
        if as_int(r, "n_prompt") == prompt_tokens and as_int(r, "n_gen") == gen_tokens:
            return as_float(r, "avg_ts", "t/s")
    return 0.0

pp_tps = find_legacy(rows, "avg_ts", f"pp{NP}") or find_legacy(rows, "t/s", f"pp{NP}")
tg_tps = find_legacy(rows, "avg_ts", f"tg{NG}") or find_legacy(rows, "t/s", f"tg{NG}")
pp_tps = pp_tps or find_current(rows, NP, 0)
tg_tps = tg_tps or find_current(rows, 0, NG)

ttft_ms = (1000.0 * NP / pp_tps) if pp_tps > 0 else 0.0
itl_ms  = (1000.0 / tg_tps) if tg_tps > 0 else 0.0

out = {
  "skill": "jetson-llm-benchmark",
  "runtime": "llama.cpp",
  "model": "${MODEL_FILE}",
  "sku": "${JETSON_SKU:-unknown}",
  "generation": "${JETSON_GENERATION:-unknown}",
  "product_line": "${JETSON_PRODUCT_LINE:-unknown}",
  "variant": "${JETSON_VARIANT:-unknown}",
  "l4t": "${JETSON_L4T_VERSION:-unknown}",
  "container": "${CONTAINER}",
  "config": {"n_prompt": ${NP}, "n_gen": ${NG}, "n_gpu_layers": ${NGL}},
  "metrics": {
    "ttft_ms_p50": round(ttft_ms, 2),
    "itl_ms_p50": round(itl_ms, 2),
    "tpot_ms_p50": round(itl_ms, 2),
    "throughput_tok_s": round(tg_tps, 2)
  },
  "warnings": json.loads('''${WARN_JSON}'''),
}
print(json.dumps(out, indent=2))
PY
