# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ============================================================================
# Model/provider preflight shared by the Pi pack drivers and smoke_test.sh.
# Source it, then call (with MODEL set):
#   kit_prepare_model_env       export Pi-side nim config; register MODEL's nim id
#   kit_check_model_key <tag>   return 1 (after printing why) if MODEL is not
#                               set, if a nim/ MODEL has no endpoint, or if the
#                               provider's credential variable is not exported
# MODEL and the nim/ endpoint have no default: the user names both.
# Only tests whether a credential variable is non-empty; never prints values.
# ============================================================================

KIT_NIM_DEFAULT_KEY_VAR=NVIDIA_INFERENCE_API_KEY

kit_nim_key_var() {
  local var=${PI_KIT_NIM_API_KEY_VAR:-$KIT_NIM_DEFAULT_KEY_VAR}
  [[ $var =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || var=$KIT_NIM_DEFAULT_KEY_VAR
  echo "$var"
}

# Endpoint of the kit's nim/ provider; adapters/pi/nvidia-provider.ts reads the
# same two variables. Empty = not configured.
kit_nim_base_url() {
  local url=${PI_KIT_NIM_BASE_URL:-${NVIDIA_INFERENCE_BASE_URL:-}}
  echo "${url//[[:space:]]/}"
}

# Credential variable for a <provider>/<model> ref; empty = not a known Pi
# built-in (models.json providers, keyless local servers).
kit_model_key_var() {
  case "$1" in
    nim/*)         kit_nim_key_var ;;
    nvidia/*)      echo NVIDIA_API_KEY ;;
    anthropic/*)   echo ANTHROPIC_API_KEY ;;
    openai/*)      echo OPENAI_API_KEY ;;
    google/*)      echo GEMINI_API_KEY ;;
    openrouter/*)  echo OPENROUTER_API_KEY ;;
    deepseek/*)    echo DEEPSEEK_API_KEY ;;
    mistral/*)     echo MISTRAL_API_KEY ;;
    groq/*)        echo GROQ_API_KEY ;;
    cerebras/*)    echo CEREBRAS_API_KEY ;;
    xai/*)         echo XAI_API_KEY ;;
    fireworks/*)   echo FIREWORKS_API_KEY ;;
    together/*)    echo TOGETHER_API_KEY ;;
    huggingface/*) echo HF_TOKEN ;;
    *)             echo "" ;;
  esac
}

kit_prepare_model_env() {
  local v id
  # kit.env is sourced without export; these are read by nvidia-provider.ts.
  for v in PI_KIT_NIM_BASE_URL NVIDIA_INFERENCE_BASE_URL PI_KIT_NIM_API_KEY_VAR \
           PI_KIT_NIM_MODELS PI_KIT_NIM_CONTEXT_WINDOW PI_KIT_NIM_MAX_TOKENS; do
    [ -n "${!v:-}" ] && export "$v"
  done
  case "${MODEL:-}" in
    nim/*)
      id=${MODEL#nim/}
      case "$id" in *:off|*:minimal|*:low|*:medium|*:high|*:xhigh) id=${id%:*} ;; esac
      export PI_KIT_NIM_MODELS="${PI_KIT_NIM_MODELS:+$PI_KIT_NIM_MODELS,}$id"
      ;;
  esac
}

kit_check_model_key() {
  local tag=$1 var
  case "${MODEL:-}" in
    "") echo "$tag ABORT: set MODEL (export it or set it in ~/.tao-kit/kit.env) to <provider>/<model-id>[:thinking], the model that executes the cards, e.g. nvidia/<model-id> or nim/<model-id>:off; there is no default model" >&2; return 1 ;;
    ?*/?*) ;;
    *) echo "$tag ABORT: MODEL must be <provider>/<model-id>, got '${MODEL:-}'" >&2; return 1 ;;
  esac
  case "$MODEL" in
    nim/*)
      if [ -z "$(kit_nim_base_url)" ]; then
        echo "$tag ABORT: MODEL=$MODEL uses the kit's nim/ provider, which has no default endpoint: set PI_KIT_NIM_BASE_URL (export it or set it in ~/.tao-kit/kit.env) to the base URL, including /v1, of the OpenAI-compatible endpoint that serves this model, e.g. http://localhost:8000/v1" >&2
        return 1
      fi ;;
  esac
  var=$(kit_model_key_var "$MODEL")
  if [ -z "$var" ]; then
    echo "$tag key preflight skipped for provider '${MODEL%%/*}' (not a Pi built-in; Pi resolves it from ~/.pi/agent/models.json)" >&2
    return 0
  fi
  if [ -n "${PI_KIT_SKIP_KEY_PREFLIGHT:-}" ]; then
    echo "$tag key preflight skipped for $var (PI_KIT_SKIP_KEY_PREFLIGHT is set)" >&2
    return 0
  fi
  [ -n "${!var:-}" ] && return 0
  echo "$tag ABORT: export $var for MODEL=$MODEL (or set PI_KIT_SKIP_KEY_PREFLIGHT=1 if Pi reads it from ~/.pi/agent/auth.json)" >&2
  return 1
}
