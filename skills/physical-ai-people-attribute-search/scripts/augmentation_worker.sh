#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# PAS augmentation worker: preprocess person crops into stitched pane images,
# generate per-sample augmentation configs from the distribution YAML, run
# image-edit augmentation with MCQ verification, then post-process (split
# panes back to per-view crops and build the augmented dataset JSON).

set -euo pipefail
export UV_PROJECT_ENVIRONMENT=/app/.venv

if command -v python3 &>/dev/null; then
  PY=python3
elif command -v python &>/dev/null; then
  PY=python
elif [ -x /app/.venv/bin/python ]; then
  PY=/app/.venv/bin/python
else
  echo "ERROR: No Python interpreter found" >&2; exit 1
fi
export PY

set -a; source "${SETUP_DIR}/.env"; set +a

export OPENAI_API_KEY="${OPENAI_API_KEY:-${VLM_API_KEY:-${LLM_API_KEY:-${NVIDIA_API_KEY:-${NGC_CLI_API_KEY:-}}}}}"
export VLM_API_KEY="${VLM_API_KEY:-${OPENAI_API_KEY:-${NVIDIA_API_KEY:-${NGC_CLI_API_KEY:-}}}}"
export LLM_API_KEY="${LLM_API_KEY:-${OPENAI_API_KEY:-${NVIDIA_API_KEY:-${NGC_CLI_API_KEY:-}}}}"
export IMAGE_EDIT_API_KEY="${IMAGE_EDIT_API_KEY:-${OPENAI_API_KEY:-${NVIDIA_API_KEY:-${NGC_CLI_API_KEY:-}}}}"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/endpoint_common.sh"

_AUTH_HDR="$(make_auth_header "${VLM_API_KEY:-}")"
_LLM_AUTH_HDR="$(make_auth_header "${LLM_API_KEY:-}")"
_IE_AUTH_HDR="$(make_auth_header "${IMAGE_EDIT_API_KEY:-}")"

_ORIG_IMAGE_EDIT_URL="${IMAGE_EDIT_URL}"
_ORIG_VLM_URL="${VLM_URL}"
_ORIG_LLM_URL="${LLM_URL}"
IMAGE_EDIT_URL="$(default_openai_base_url "${IMAGE_EDIT_URL}")"
VLM_URL="$(default_openai_base_url "${VLM_URL}")"
LLM_URL="$(default_openai_base_url "${LLM_URL}")"

if [ "${WAIT_FOR_IMAGE_EDIT:-0}" = "1" ]; then
    echo "Waiting for Image Edit server..."
    RESOLVED_ENDPOINT_URL=""
    RESOLVED_MODELS_JSON=""
    wait_for_models_ready "ImageEdit" "${_ORIG_IMAGE_EDIT_URL}" "${_IE_AUTH_HDR}"
    IMAGE_EDIT_URL="${RESOLVED_ENDPOINT_URL}"
    IMAGE_EDIT_MODEL="$(extract_first_model_id "${RESOLVED_MODELS_JSON}")"
    if [ -z "${IMAGE_EDIT_MODEL}" ]; then
        echo "ERROR: Image Edit endpoint responded but no model id found at ${IMAGE_EDIT_URL}/models" >&2
        exit 1
    fi
    echo "Image Edit ready: ${IMAGE_EDIT_MODEL} (${IMAGE_EDIT_URL})"
fi

if [ "${WAIT_FOR_VLM:-0}" = "1" ]; then
    echo "Waiting for VLM server..."
    RESOLVED_ENDPOINT_URL=""
    RESOLVED_MODELS_JSON=""
    wait_for_models_ready "VLM" "${_ORIG_VLM_URL}" "${_AUTH_HDR}"
    VLM_URL="${RESOLVED_ENDPOINT_URL}"
    VLM_MODEL="$(extract_first_model_id "${RESOLVED_MODELS_JSON}")"
    if [ -z "${VLM_MODEL}" ]; then
        echo "ERROR: VLM endpoint responded but no model id found at ${VLM_URL}/models" >&2
        exit 1
    fi
    echo "VLM ready: ${VLM_MODEL} (${VLM_URL})"
fi

if [ "${WAIT_FOR_LLM:-0}" = "1" ]; then
    echo "Waiting for LLM server..."
    RESOLVED_ENDPOINT_URL=""
    RESOLVED_MODELS_JSON=""
    wait_for_models_ready "LLM" "${_ORIG_LLM_URL}" "${_LLM_AUTH_HDR}"
    LLM_URL="${RESOLVED_ENDPOINT_URL}"
    LLM_MODEL="$(extract_first_model_id "${RESOLVED_MODELS_JSON}")"
    if [ -z "${LLM_MODEL}" ]; then
        echo "ERROR: LLM endpoint responded but no model id found at ${LLM_URL}/models" >&2
        exit 1
    fi
    echo "LLM ready: ${LLM_MODEL} (${LLM_URL})"
fi

cd /app

PANES_DIR="/app/data/out/panes"
AUG_CONFIGS_DIR="/app/data/out/generated_configs"
AUG_OUTPUTS_DIR="/app/data/out/augmented_outputs"
DATASET_DIR="/app/data/out/augmented_dataset"
DIST_CONFIG="${SETUP_DIR}/configs/distribution_config.yaml"
AUG_CONFIG="${SETUP_DIR}/configs/augmentation_config.yaml"

# ── Step 1: Copy input data ──
echo "=== Step 1: Copying input data ==="
mkdir -p /app/data/in
cp -r "${DATA_INPUT}"/. /app/data/in/ 2>/dev/null || true
echo "Input data staged at /app/data/in"
ls /app/data/in/ | head -20

# ── Step 2: Preprocess — combine multi-view crops into pane images ──
echo "=== Step 2: Preprocessing — combining multi-view crops into panes ==="
mkdir -p "${PANES_DIR}"
uv run python modules/data_processing/combine_panes.py \
    /app/data/in \
    "${PANES_DIR}"

PANE_COUNT=$(find "${PANES_DIR}" -maxdepth 1 -name "*.jpg" -o -name "*.png" | wc -l)
echo "Preprocessed ${PANE_COUNT} pane images"

if [ "${PANE_COUNT}" -eq 0 ]; then
    echo "ERROR: No pane images produced from input data" >&2
    exit 1
fi

# ── Step 3: Generate per-sample augmentation configs ──
echo "=== Step 3: Generating per-sample augmentation configs ==="
mkdir -p "${AUG_CONFIGS_DIR}" "${AUG_OUTPUTS_DIR}"

if [ -f "${DIST_CONFIG}" ]; then
    uv run python -c "
import yaml, os, random, itertools

with open('${DIST_CONFIG}') as f:
    dist = yaml.safe_load(f)

panes_dir = '${PANES_DIR}'
configs_dir = '${AUG_CONFIGS_DIR}'
outputs_dir = '${AUG_OUTPUTS_DIR}'
n_aug = int('${N_AUGMENTATIONS}')

variables = dist.get('variables', {})
conditional = dist.get('conditional_variables', {})

pane_files = sorted([f for f in os.listdir(panes_dir)
                     if f.endswith(('.jpg', '.png', '.jpeg'))])

config_count = 0
for pane_file in pane_files:
    person_id = os.path.splitext(pane_file)[0]
    for aug_idx in range(n_aug):
        sampled = {}
        for var_name, options in variables.items():
            if isinstance(options, dict):
                choices = list(options.keys())
                weights = list(options.values())
                sampled[var_name] = random.choices(choices, weights=weights, k=1)[0]
            elif isinstance(options, list):
                sampled[var_name] = random.choice(options)

        for var_name, cond_spec in conditional.items():
            dep = cond_spec.get('depends_on', '')
            dep_val = sampled.get(dep, '')
            dists = cond_spec.get('distributions', {})
            if dep_val in dists:
                sub = dists[dep_val]
                if isinstance(sub, dict):
                    choices = list(sub.keys())
                    weights = list(sub.values())
                    sampled[var_name] = random.choices(choices, weights=weights, k=1)[0]

        out_dir = os.path.join(outputs_dir, person_id, f'aug_{aug_idx}')
        os.makedirs(out_dir, exist_ok=True)
        cfg = {
            'person_id': person_id,
            'aug_index': aug_idx,
            'input_image': os.path.join(panes_dir, pane_file),
            'output_dir': out_dir,
            'variables': sampled,
        }
        cfg_path = os.path.join(configs_dir, f'{person_id}_aug{aug_idx}.yaml')
        with open(cfg_path, 'w') as cf:
            yaml.dump(cfg, cf, default_flow_style=False)
        config_count += 1

print(f'Generated {config_count} augmentation configs')
"
else
    echo "WARNING: distribution_config.yaml not found — using container defaults"
fi

# ── Step 4: Run image-edit augmentation ──
echo "=== Step 4: Running image-edit augmentation ==="
TOTAL=0
PASSED=0
FAILED=0

for cfg_file in "${AUG_CONFIGS_DIR}"/*.yaml; do
    [ -f "${cfg_file}" ] || continue
    TOTAL=$((TOTAL + 1))

    PERSON_ID=$(${PY} -c "import yaml; print(yaml.safe_load(open('${cfg_file}'))['person_id'])")
    AUG_IDX=$(${PY} -c "import yaml; print(yaml.safe_load(open('${cfg_file}'))['aug_index'])")
    INPUT_IMG=$(${PY} -c "import yaml; print(yaml.safe_load(open('${cfg_file}'))['input_image'])")
    OUT_SUBDIR=$(${PY} -c "import yaml; print(yaml.safe_load(open('${cfg_file}'))['output_dir'])")

    echo "--- Augmenting ${PERSON_ID} aug_${AUG_IDX} ---"

    VARIABLES_YAML=$(${PY} -c "
import yaml
cfg = yaml.safe_load(open('${cfg_file}'))
v = cfg.get('variables', {})
parts = []
for k, val in v.items():
    parts.append(f\"'captioning.llm.variables.{k}=[{val}]'\")
print(' '.join(parts))
")

    AUG_EXIT=0
    eval uv run modules/cli.py --config "${AUG_CONFIG}" \
        "data.0.inputs.rgb=${INPUT_IMG}" \
        "data.0.output.video=${OUT_SUBDIR}/output.jpg" \
        "data.0.output.caption=${OUT_SUBDIR}/output.txt" \
        "data.0.output.metadata=${OUT_SUBDIR}/output_metadata.json" \
        "endpoints.image_edit.url=${IMAGE_EDIT_URL}" \
        "endpoints.image_edit.model=${IMAGE_EDIT_MODEL:-Qwen/Qwen-Image-Edit-2511}" \
        "endpoints.image_edit.timeout=300" \
        "endpoints.vlm.url=${VLM_URL}" \
        "endpoints.vlm.model=${VLM_MODEL:-qwen3-vl}" \
        "endpoints.vlm.timeout=300" \
        "endpoints.llm.url=${LLM_URL}" \
        "endpoints.llm.model=${LLM_MODEL:-qwen25-14b}" \
        ${VARIABLES_YAML} || AUG_EXIT=$?

    if [ "${AUG_EXIT}" -ne 0 ]; then
        echo "WARNING: Augmentation failed for ${PERSON_ID} aug_${AUG_IDX} (exit ${AUG_EXIT})"
        FAILED=$((FAILED + 1))
    elif [ -f "${OUT_SUBDIR}/output.jpg" ]; then
        PASSED=$((PASSED + 1))
    else
        echo "WARNING: No output produced for ${PERSON_ID} aug_${AUG_IDX}"
        FAILED=$((FAILED + 1))
    fi
done

echo "=== Augmentation summary: total=${TOTAL} passed=${PASSED} failed=${FAILED} ==="

# ── Step 5: Post-process — split panes back to per-view crops ──
echo "=== Step 5: Post-processing — building augmented dataset ==="
mkdir -p "${DATASET_DIR}"

if command -v uv >/dev/null 2>&1; then
    uv run --no-sync python modules/data_processing/create_PAS_augmented_dataset.py \
        --base-dir "${PANES_DIR}" \
        --augmented-folders "${AUG_OUTPUTS_DIR}" \
        --output-dir "${DATASET_DIR}" \
        --output-json augmented_data.json || echo "WARNING: Post-processing script failed; raw outputs still available"
fi

# ── Step 6: Copy outputs to OSMO output URL ──
echo "=== Step 6: Staging outputs ==="
cp -r "${AUG_OUTPUTS_DIR}"/. "${OUTPUT_DIR}/" 2>/dev/null || true
if [ -d "${DATASET_DIR}" ]; then
    cp -r "${DATASET_DIR}"/. "${OUTPUT_DIR}/dataset/" 2>/dev/null || true
fi

echo "=== augmentation_worker complete ==="
