#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# PAS preflight checks:
#   - optional NGC key discovery/refresh for nvcr_io credential maintenance
#   - optional outbound probes (workflow image registry access + HF)
#   - ensures required OSMO credentials exist (hf_token always; nvcr_io when key provided)
#   - creates missing credentials from env vars; refreshes existing ones on --refresh
# NOTE:
#   This script does NOT validate credentials for external VLM/LLM/Image-Edit endpoints.
#   Endpoint API keys/tokens must be validated separately per endpoint.
#
# Usage:
#   preflight_credentials.sh [--no-probe] [--workflow <workflow-yaml>] [--refresh|--overwrite]
#
# Exit 0 when all checks pass, else exit 1 with remediation.

set -euo pipefail

usage() {
  echo "usage: $0 [--no-probe] [--workflow <workflow-yaml>] [--refresh|--overwrite]" >&2
  exit 2
}

probe=true
workflow_file=""
registry_probe_checked=false
refresh_credentials=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-probe)
      probe=false
      shift
      ;;
    --workflow)
      [[ $# -ge 2 ]] || usage
      workflow_file="$2"
      shift 2
      ;;
    --workflow=*)
      workflow_file="${1#--workflow=}"
      shift
      ;;
    --refresh|--overwrite)
      refresh_credentials=true
      shift
      ;;
    *)
      usage
      ;;
  esac
done

if [ -n "${workflow_file}" ] && [ ! -f "${workflow_file}" ]; then
  echo "workflow file not found: ${workflow_file}" >&2
  usage
fi

user_supplied_ngc_key=false
for var_name in NGC_API_KEY NGC_CLI_API_KEY NVIDIA_API_KEY OPENAI_API_KEY VLM_API_KEY LLM_API_KEY IMAGE_EDIT_API_KEY; do
  if [[ -n "${!var_name:-}" ]]; then
    user_supplied_ngc_key=true
    break
  fi
done

user_supplied_hf_token=false
for var_name in HF_TOKEN HUGGING_FACE_HUB_TOKEN; do
  if [[ -n "${!var_name:-}" ]]; then
    user_supplied_hf_token=true
    break
  fi
done

emit_user_input_required() {
  local msg="${1:-Missing required user input.}"
  echo "USER_INPUT_REQUIRED: ${msg}" >&2
}

resolve_ngc_api_key() {
  local candidate=""
  local var_name=""

  for var_name in NGC_API_KEY NGC_CLI_API_KEY NVIDIA_API_KEY OPENAI_API_KEY VLM_API_KEY LLM_API_KEY IMAGE_EDIT_API_KEY; do
    candidate="${!var_name:-}"
    case "${candidate}" in
      "Authorization: Bearer "*) candidate="${candidate#Authorization: Bearer }" ;;
      "Bearer "*) candidate="${candidate#Bearer }" ;;
    esac
    if [[ "${candidate}" =~ ^[Nn][Vv][Aa][Pp][Ii]- ]]; then
      printf '%s' "${candidate}"
      return 0
    fi
  done

  for var_name in NGC_API_KEY NGC_CLI_API_KEY; do
    candidate="${!var_name:-}"
    case "${candidate}" in
      "Authorization: Bearer "*) candidate="${candidate#Authorization: Bearer }" ;;
      "Bearer "*) candidate="${candidate#Bearer }" ;;
    esac
    if [ -n "${candidate}" ]; then
      printf '%s' "${candidate}"
      return 0
    fi
  done

  return 0
}

resolve_hf_token() {
  local env_value="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
  local discovered=""
  local token_file="${HF_TOKEN_FILE:-${HOME}/.cache/huggingface/token}"
  if [ -n "${env_value}" ]; then
    printf '%s' "${env_value}"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    discovered="$(python3 - <<'PY'
try:
    from huggingface_hub import get_token
    t = get_token() or ""
    print(t)
except Exception:
    pass
PY
)"
    if [ -n "${discovered}" ]; then
      printf '%s' "${discovered}"
      return 0
    fi
  fi
  if [ -f "${token_file}" ]; then
    local first_line=""
    if IFS= read -r first_line < "${token_file}"; then
      printf '%s' "${first_line}"
    fi
  fi
  return 0
}

extract_workflow_nvcr_images() {
  local workflow="$1"
  awk '
    /^[[:space:]]*image:[[:space:]]*/ {
      line=$0
      sub(/^[[:space:]]*image:[[:space:]]*/, "", line)
      sub(/[[:space:]]+#.*/, "", line)
      gsub(/["'"'"'"]/, "", line)
      if (line ~ /^nvcr\.io\//) {
        print line
      }
    }
  ' "${workflow}" | sort -u
}

probe_nvcr_image_ref() {
  local image_ref="$1"
  local without_host="${image_ref#nvcr.io/}"
  local repo="${without_host}"
  local ref="latest"
  local manifest_url=""

  if [[ "${without_host}" == *@* ]]; then
    repo="${without_host%@*}"
    ref="${without_host#*@}"
  elif [[ "${without_host}" == *:* ]]; then
    repo="${without_host%:*}"
    ref="${without_host##*:}"
  fi

  manifest_url="https://nvcr.io/v2/${repo}/manifests/${ref}"

  local status
  status="$(curl -sS -o /dev/null -w '%{http_code}' \
    -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
    "${manifest_url}")" || {
    echo "000"
    return
  }
  echo "${status}"
}

run_workflow_registry_probe() {
  local workflow="$1"
  local image_refs=""
  local image_ref=""
  local status=""
  local ok_count=0
  local -a denied_refs=()
  local -a missing_refs=()

  if [ ! -f "${workflow}" ]; then
    echo "Workflow image probe skipped: workflow file not found: ${workflow}" >&2
    return 0
  fi

  image_refs="$(extract_workflow_nvcr_images "${workflow}")"
  if [ -z "${image_refs}" ]; then
    echo "Workflow image probe skipped: no nvcr.io images found in ${workflow}" >&2
    return 0
  fi

  echo "Probing nvcr.io access for workflow images in ${workflow}:" >&2
  while IFS= read -r image_ref; do
    [ -n "${image_ref}" ] || continue
    status="$(probe_nvcr_image_ref "${image_ref}")"
    case "${status}" in
      200)
        echo "  OK registry access: ${image_ref}" >&2
        ok_count=$((ok_count + 1))
        ;;
      401|403)
        echo "  FAIL registry access denied (HTTP ${status}): ${image_ref}" >&2
        denied_refs+=("${image_ref} (HTTP ${status})")
        ;;
      404)
        echo "  FAIL registry image ref not found (HTTP 404): ${image_ref}" >&2
        missing_refs+=("${image_ref} (HTTP 404)")
        ;;
      *)
        echo "  WARN registry probe returned HTTP ${status}: ${image_ref}" >&2
        ;;
    esac
  done <<< "${image_refs}"

  if [[ "${missing_refs[0]+__set__}" == "__set__" ]]; then
    echo "NGC registry probe found missing/unpublished workflow image refs:" >&2
    printf '  - %s\n' "${missing_refs[@]}" >&2
    return 1
  fi

  if [[ "${denied_refs[0]+__set__}" == "__set__" ]]; then
    echo "NGC registry probe reported HTTP 401/403 on workflow image refs:" >&2
    printf '  - %s\n' "${denied_refs[@]}" >&2
    return 1
  fi

  if [ "${ok_count}" -gt 0 ]; then
    registry_probe_checked=true
  fi
  return 0
}

# 1) Check existing OSMO credentials
present=$(osmo credential list | awk 'NR>1 {print $1}' | sort -u)
need_hf=false
grep -qx 'hf_token' <<<"$present" || need_hf=true

# 2) Discover NGC key
if [ -z "${NGC_API_KEY:-}" ]; then
  discovered_ngc_api_key="$(resolve_ngc_api_key)"
  if [ -n "${discovered_ngc_api_key}" ]; then
    export NGC_API_KEY="${discovered_ngc_api_key}"
    echo "AUTO_SECRET_LOADED: NGC API key discovered from environment." >&2
  fi
fi
if [ -z "${HF_TOKEN:-}" ]; then
  discovered_hf_token="$(resolve_hf_token)"
  if [ -n "${discovered_hf_token}" ]; then
    export HF_TOKEN="${discovered_hf_token}"
    echo "AUTO_SECRET_LOADED: HF token discovered from local cache." >&2
  fi
fi

# 3) Check required secrets
missing_env=()
if $need_hf && [[ -z "${HF_TOKEN:-}" ]]; then
  missing_env+=(HF_TOKEN)
fi
if [[ "${missing_env[0]+__set__}" == "__set__" ]]; then
  echo "Missing required secrets to create absent OSMO credentials:" >&2
  printf '  - %s\n' "${missing_env[@]}" >&2
  emit_user_input_required "Provide missing secrets for absent credentials: ${missing_env[*]}"
  exit 1
fi

# 4) Workflow image registry probe
if $probe && [ -n "${workflow_file}" ]; then
  run_workflow_registry_probe "${workflow_file}" || exit 1
elif $probe && [ -z "${workflow_file}" ]; then
  echo "Workflow image probe skipped: pass --workflow <workflow-yaml> to validate." >&2
fi

# 5) Ensure required OSMO credentials
if ! grep -qx 'nvcr_io' <<<"$present"; then
  if [[ -n "${NGC_API_KEY:-}" ]]; then
    echo ">>> setting OSMO credential nvcr_io from NGC_API_KEY" >&2
    osmo credential set nvcr_io --type REGISTRY \
      --payload registry=nvcr.io username='$oauthtoken' auth="$NGC_API_KEY"
  else
    echo ">>> nvcr_io credential missing, continuing without it (public nvcr.io pulls expected)." >&2
  fi
elif $refresh_credentials || $user_supplied_ngc_key; then
  if [[ -n "${NGC_API_KEY:-}" ]]; then
    echo ">>> refreshing existing OSMO credential nvcr_io" >&2
    osmo credential set nvcr_io --type REGISTRY \
      --payload registry=nvcr.io username='$oauthtoken' auth="$NGC_API_KEY"
  fi
fi

if ! grep -qx 'hf_token' <<<"$present"; then
  echo ">>> setting OSMO credential hf_token from HF_TOKEN" >&2
  osmo credential set hf_token --type GENERIC \
    --payload token="$HF_TOKEN" HF_TOKEN="$HF_TOKEN"
elif $refresh_credentials || $user_supplied_hf_token; then
  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo ">>> refreshing existing OSMO credential hf_token" >&2
    osmo credential set hf_token --type GENERIC \
      --payload token="$HF_TOKEN" HF_TOKEN="$HF_TOKEN"
  fi
fi

present_after=$(osmo credential list | awk 'NR>1 {print $1}' | sort -u)
missing_after=()
for name in hf_token; do
  grep -qx "$name" <<<"$present_after" || missing_after+=("$name")
done
if [[ "${missing_after[0]+__set__}" == "__set__" ]]; then
  echo "OSMO credentials still missing after auto-set:" >&2
  printf '  - %s\n' "${missing_after[@]}" >&2
  exit 1
fi

# 6) OSMO control-plane readiness
pool_status=$(osmo pool list --mode free 2>&1) || {
  echo "OSMO pool query failed." >&2
  exit 1
}
if ! grep -Eqi 'online' <<<"$pool_status"; then
  echo "No ONLINE pool found in osmo pool list --mode free output." >&2
  exit 1
fi

echo "OK: required secrets valid; OSMO hf_token credential present."
if grep -qx 'nvcr_io' <<<"$present_after"; then
  echo "NOTE: nvcr_io credential is present."
else
  echo "NOTE: nvcr_io credential is absent; public nvcr.io pulls are expected."
fi
echo "NOTE: external endpoint credentials (VLM/LLM/Image-Edit API keys) are not validated by this script."
