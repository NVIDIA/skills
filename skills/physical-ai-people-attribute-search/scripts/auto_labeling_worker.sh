#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# PAS auto-labeling worker: generate person-attribute captions and synonymous
# search queries for each augmented image using the shipped person_attributes
# question bank inside the paidf-auto-labeling container.

set -euo pipefail
export UV_PROJECT_ENVIRONMENT=/opt/venv
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
set -a; source "${SETUP_DIR}/.env"; set +a

export OPENAI_API_KEY="${OPENAI_API_KEY:-${VLM_API_KEY:-${LLM_API_KEY:-${NVIDIA_API_KEY:-${NGC_CLI_API_KEY:-}}}}}"
export VLM_API_KEY="${VLM_API_KEY:-${OPENAI_API_KEY:-${NVIDIA_API_KEY:-${NGC_CLI_API_KEY:-}}}}"
export LLM_API_KEY="${LLM_API_KEY:-${OPENAI_API_KEY:-${NVIDIA_API_KEY:-${NGC_CLI_API_KEY:-}}}}"
export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/endpoint_common.sh"

_AUTH_HDR="$(make_auth_header "${VLM_API_KEY:-}")"
_LLM_AUTH_HDR="$(make_auth_header "${LLM_API_KEY:-}")"

_ORIG_VLM_URL="${VLM_URL}"
_ORIG_LLM_URL="${LLM_URL}"
VLM_URL="$(default_openai_base_url "${VLM_URL}")"
LLM_URL="$(default_openai_base_url "${LLM_URL}")"

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

cd /workspace
if [ -f docker/entrypoint.sh ]; then bash docker/entrypoint.sh; fi

QUESTION_BANK="/workspace/cookbooks/person_attributes/question_bank.json"
VLM_SCENE_PROMPT="/workspace/cookbooks/person_attributes/prompts/mcq/question_driven_vlm_llm/vlm_scene_prompt_template.md"

if [ ! -f "${QUESTION_BANK}" ]; then
    echo "ERROR: person_attributes question bank not found at ${QUESTION_BANK}" >&2
    exit 1
fi

_arr_len() { echo "$#"; }

IMAGE_DIRS=()
if [ -d "${AUGMENTED_INPUT}/dataset/augmented_imgs" ]; then
    for d in "${AUGMENTED_INPUT}/dataset/augmented_imgs"/*/; do
        [ -d "$d" ] && IMAGE_DIRS+=("$d")
    done
fi

if [ "$(_arr_len "${IMAGE_DIRS[@]+"${IMAGE_DIRS[@]}"}")" -eq 0 ]; then
    while IFS= read -r img; do
        [ -n "${img}" ] && IMAGE_DIRS+=("$(dirname "${img}")")
    done < <(find "${AUGMENTED_INPUT}" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) -name "output.*" 2>/dev/null)
fi

if [ "$(_arr_len "${IMAGE_DIRS[@]+"${IMAGE_DIRS[@]}"}")" -eq 0 ]; then
    while IFS= read -r img; do
        [ -n "${img}" ] && IMAGE_DIRS+=("${img}")
    done < <(find "${AUGMENTED_INPUT}" -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) 2>/dev/null | head -100)
fi

_N_DIRS=$(_arr_len "${IMAGE_DIRS[@]+"${IMAGE_DIRS[@]}"}")
echo "Found ${_N_DIRS} images/directories to caption"
if [ "${_N_DIRS}" -eq 0 ]; then
    echo "ERROR: No augmented images found in ${AUGMENTED_INPUT}" >&2
    exit 1
fi

TOTAL=0
PASSED=0
FAILED=0

# Process unique directories or individual images
declare -A SEEN_DIRS
for item in "${IMAGE_DIRS[@]}"; do
    img_path=""
    if [ -d "${item}" ]; then
        [ -n "${SEEN_DIRS[${item}]+_}" ] && continue
        SEEN_DIRS["${item}"]=1
        img_path=$(find "${item}" -maxdepth 1 -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) -print -quit 2>/dev/null)
    elif [ -f "${item}" ]; then
        img_path="${item}"
    fi
    [ -z "${img_path}" ] && continue

    TOTAL=$((TOTAL + 1))
    RUN_NAME="caption_$(basename "$(dirname "${img_path}")")_$(basename "${img_path}" | sed 's/\.[^.]*$//')"
    AL_OUT="${OUTPUT_DIR}/${RUN_NAME}"
    mkdir -p "${AL_OUT}"

    echo "--- Captioning: ${img_path} -> ${RUN_NAME} ---"

    AL_EXIT=0
    uv run python modules/cli.py \
        --config configs/pipeline_example.yaml \
        super_resolution.enabled=false \
        detection_and_tracking.enabled=false \
        vlm_json.enabled=false \
        mcq_generation.enabled=true \
        mcq_generation.mode=question-driven-vlm-llm \
        mcq_generation.window_metadata_extraction.single_window=true \
        mcq_generation.window_metadata_extraction.vlm_verify_enabled=false \
        "mcq_generation.window_metadata_extraction.question_bank_file=${QUESTION_BANK}" \
        "mcq_generation.window_metadata_extraction.qd_vlm_scene_prompt_template_file=${VLM_SCENE_PROMPT}" \
        "data.0.inputs.video_path=${img_path}" \
        "data.0.output.out_dir=${AL_OUT}" \
        "endpoints.vlm.url=${VLM_URL}" \
        "endpoints.vlm.model=${VLM_MODEL:-qwen3-vl}" \
        "endpoints.llm.url=${LLM_URL}" \
        "endpoints.llm.model=${LLM_MODEL:-qwen25-14b}" || AL_EXIT=$?

    if [ "${AL_EXIT}" -ne 0 ]; then
        echo "WARNING: Auto-labeling failed for ${img_path} (exit ${AL_EXIT})"
        FAILED=$((FAILED + 1))
    elif [ -f "${AL_OUT}/task/open_qa.json" ]; then
        PASSED=$((PASSED + 1))
    else
        echo "WARNING: No open_qa.json produced for ${img_path}"
        FAILED=$((FAILED + 1))
    fi
done

echo "=== Auto-labeling summary: total=${TOTAL} passed=${PASSED} failed=${FAILED} ==="
echo "=== auto_labeling_worker complete ==="
