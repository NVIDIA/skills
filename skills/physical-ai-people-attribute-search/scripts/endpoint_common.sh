#!/usr/bin/env bash
# Shared endpoint + worker utilities for PAS scripts.
# Based on the VDA helpers (model-agnostic, OpenAI-compatible endpoints), plus a
# PAS-specific /v1/health/ready fallback for Triton-based Visual GenAI NIMs that
# do not expose an OpenAI /v1/models listing.

make_auth_header() {
    local token="${1:-}"
    if [ -z "${token}" ]; then
        echo ""
        return
    fi
    if [[ "${token}" == Authorization:* ]]; then
        echo "${token}"
    elif [[ "${token}" == Bearer* ]]; then
        echo "Authorization: ${token}"
    else
        echo "Authorization: Bearer ${token}"
    fi
}

ensure_scheme_url() {
    local raw="${1:-}"
    if [ -z "${raw}" ]; then
        echo ""
        return
    fi
    if [[ "${raw}" == http://* || "${raw}" == https://* ]]; then
        echo "${raw}"
    else
        echo "http://${raw}"
    fi
}

strip_query_fragment() {
    local raw="${1:-}"
    raw="${raw%%\?*}"
    raw="${raw%%\#*}"
    echo "${raw}"
}

openai_candidate_seeds() {
    local raw
    raw="$(ensure_scheme_url "${1:-}")"
    raw="$(strip_query_fragment "${raw}")"
    raw="${raw%/}"
    if [ -z "${raw}" ]; then
        return
    fi

    local changed=1
    while [ "${changed}" -eq 1 ]; do
        changed=0
        case "${raw}" in
            */v1/chat/completions) raw="${raw%/v1/chat/completions}"; changed=1 ;;
            */chat/completions) raw="${raw%/chat/completions}"; changed=1 ;;
            */v1/completions) raw="${raw%/v1/completions}"; changed=1 ;;
            */completions) raw="${raw%/completions}"; changed=1 ;;
            */v1/responses) raw="${raw%/v1/responses}"; changed=1 ;;
            */responses) raw="${raw%/responses}"; changed=1 ;;
            */v1/embeddings) raw="${raw%/v1/embeddings}"; changed=1 ;;
            */embeddings) raw="${raw%/embeddings}"; changed=1 ;;
            */v1/models) raw="${raw%/v1/models}"; changed=1 ;;
            */models) raw="${raw%/models}"; changed=1 ;;
        esac
    done

    echo "${raw}"

    local without_scheme="${raw#*://}"
    local host="${without_scheme%%/*}"
    local scheme="${raw%%://*}"
    if [ -n "${host}" ] && [ -n "${scheme}" ]; then
        echo "${scheme}://${host}"
    fi
}

candidate_openai_base_urls() {
    local seed
    local -a seeds=()
    while IFS= read -r seed; do
        [ -n "${seed}" ] && seeds+=("${seed}")
    done < <(openai_candidate_seeds "${1:-}")

    local seen=" "
    local base
    for base in "${seeds[@]}"; do
        [ -n "${base}" ] || continue

        case " ${seen} " in
            *" ${base} "*) ;;
            *)
                echo "${base}"
                seen="${seen}${base} "
                ;;
        esac

        local alt=""
        if [[ "${base}" == */v1 ]]; then
            alt="${base%/v1}"
        else
            alt="${base}/v1"
        fi

        case " ${seen} " in
            *" ${alt} "*) ;;
            *)
                echo "${alt}"
                seen="${seen}${alt} "
                ;;
        esac
    done
}

default_openai_base_url() {
    local raw="${1:-}"
    local first=""
    local c
    while IFS= read -r c; do
        [ -n "${first}" ] || first="${c}"
        if [[ "${c}" == */v1 ]]; then
            echo "${c}"
            return
        fi
    done < <(candidate_openai_base_urls "${raw}")
    echo "${first}"
}

probe_models_json() {
    local base_url="$1"
    local auth_header="${2:-}"
    local models_url="${base_url%/}/models"
    local connect_timeout_s="${ENDPOINT_CURL_CONNECT_TIMEOUT_SECONDS:-5}"
    local max_time_s="${ENDPOINT_CURL_MAX_TIME_SECONDS:-15}"
    local response=""

    if [ -n "${auth_header}" ]; then
        if ! response=$(curl -fsS --connect-timeout "${connect_timeout_s}" --max-time "${max_time_s}" -H "${auth_header}" "${models_url}" 2>/dev/null); then
            return 1
        fi
    else
        if ! response=$(curl -fsS --connect-timeout "${connect_timeout_s}" --max-time "${max_time_s}" "${models_url}" 2>/dev/null); then
            return 1
        fi
    fi

    if printf '%s' "${response}" | grep -q '"data"'; then
        RESOLVED_MODELS_JSON="${response}"
        return 0
    fi
    return 1
}

# Probe the /v1/health/ready endpoint (used by Visual GenAI NIMs that
# do not expose an OpenAI-compatible /models listing).  Returns 0 when the
# response contains '"status":"ready"', matching both Triton and model-free-nim
# health formats.  Also populates RESOLVED_MODELS_JSON with a synthetic
# model-entry payload so callers that extract the model id (extract_first_model_id)
# get a usable value.
probe_health_ready() {
    local base_url="$1"
    local health_url="${base_url%/}/health/ready"
    local connect_timeout_s="${ENDPOINT_CURL_CONNECT_TIMEOUT_SECONDS:-5}"
    local max_time_s="${ENDPOINT_CURL_MAX_TIME_SECONDS:-15}"
    local response=""

    if ! response=$(curl -fsS --connect-timeout "${connect_timeout_s}" --max-time "${max_time_s}" "${health_url}" 2>/dev/null); then
        return 1
    fi

    if printf '%s' "${response}" | grep -qE '"ready"|"status":"ready"'; then
        # Populate a synthetic model list entry so extract_first_model_id() returns
        # a non-empty string.  Use the raw /v1 URL as the synthetic model id.
        RESOLVED_MODELS_JSON='{"object":"list","data":[{"id":"'"${base_url}"'"}]}'
        return 0
    fi
    return 1
}

wait_for_models_ready() {
    local name="$1"
    local raw_url="$2"
    local auth_header="${3:-}"
    local timeout_s="${ENDPOINT_WAIT_TIMEOUT_SECONDS:-180}"
    local interval_s="${ENDPOINT_WAIT_INTERVAL_SECONDS:-10}"
    local max_attempts=$(( timeout_s / interval_s ))
    if [ "${max_attempts}" -lt 1 ]; then max_attempts=1; fi

    local -a candidates=()
    local c
    while IFS= read -r c; do
        [ -n "${c}" ] && candidates+=("${c}")
    done < <(candidate_openai_base_urls "${raw_url}")

    if [ "${candidates[0]+__set__}" != "__set__" ]; then
        echo "ERROR: ${name} endpoint URL is empty or invalid: ${raw_url}" >&2
        return 1
    fi

    local attempt candidate
    for ((attempt=1; attempt<=max_attempts; attempt++)); do
        for candidate in "${candidates[@]}"; do
            # Try OpenAI-compatible /models probe first (most NIMs).  Falls back to
            # /v1/health/ready for Triton-based Visual GenAI NIMs that don't expose /models.
            if probe_models_json "${candidate}" "${auth_header}"; then
                RESOLVED_ENDPOINT_URL="${candidate}"
                return 0
            elif probe_health_ready "${candidate}"; then
                RESOLVED_ENDPOINT_URL="${candidate}"
                return 0
            fi
        done
        echo "Waiting for ${name} server (${attempt}/${max_attempts}): tried ${candidates[*]}" >&2
        sleep "${interval_s}"
    done

    echo "ERROR: ${name} endpoint not ready after ${timeout_s}s (tried ${candidates[*]})." >&2
    echo "Hint: provide an OpenAI-compatible base URL or invoke URL (NIM/NVCF examples accepted)." >&2
    return 1
}

extract_first_model_id() {
    local payload="${1:-}"
    printf '%s' "${payload}" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4
}
