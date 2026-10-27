#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Report GPU, container-runtime, and storage readiness on the execution host.
#
# Default probes inspect host state. Container tests are opt-in and may pull images;
# callers must establish probe consent before enabling those command-line flags.

set -o pipefail

DEFAULT_CUDA_TEST_IMAGE="nvidia/cuda:13.3.1-base-ubuntu22.04"
MIN_DOCKER_MAJOR=20
MIN_DOCKER_MINOR=10
MIN_GPU_MEMORY_GB=16
LOW_STORAGE_WARNING_GB=100
SUMMARY_MAX_CHARS=300
DEFAULT_TIMEOUT_SECONDS=30
TIMEOUT_KILL_AFTER_SECONDS=1
SCHEMA_VERSION=1
EXAMPLE_PARABRICKS_VERSION="4.8.0-1"
PROG=$(basename "$0")

usage() {
  cat <<EOF
usage: $PROG [-h] [--path PATH] [--run-container-check] [--cuda-test-image IMAGE]
       [--run-parabricks-check] [--parabricks-version TAG | --parabricks-image IMAGE]
       [--format {text,json}] [--timeout SECONDS]

Check local NVIDIA Parabricks runtime readiness and print a report.

options:
  -h, --help                Show this help message and exit.
  --path PATH               Input, output, or temporary path to include in storage checks. Can be repeated.
  --run-container-check     Run a Docker CUDA container with --gpus all to test NVIDIA Container Toolkit.
  --cuda-test-image IMAGE   CUDA image with an explicit tag or SHA-256 digest. Default: $DEFAULT_CUDA_TEST_IMAGE
  --run-parabricks-check    Run pbrun --help in the selected Parabricks container; may pull the image.
  --parabricks-version TAG  Select a Parabricks tag, for example $EXAMPLE_PARABRICKS_VERSION. Does not launch a container by itself.
  --parabricks-image IMAGE  Select a full Parabricks image with an explicit tag or SHA-256 digest instead of --parabricks-version.
  --format {text,json}      Report output format.
  --timeout SECONDS         Positive timeout for each probe (requires GNU timeout or gtimeout).
EOF
}

die_usage() {
  usage >&2
  printf '%s: error: %s\n' "$PROG" "$1" >&2
  exit 2
}

validate_tag() {
  local tag=$1 option=$2
  if [[ ! "$tag" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}$ ]] || [ "$tag" = latest ]; then
    die_usage "$option requires an explicit release tag, not latest"
  fi
}

validate_image_reference() {
  local image=$1 option=$2 repository digest
  case $image in
    *@sha256:*)
      repository=${image%@sha256:*}
      digest=${image##*@sha256:}
      [[ "$digest" =~ ^[a-f0-9]{64}$ ]] || die_usage "$option requires a 64-character SHA-256 digest"
      ;;
    *)
      # A registry port is not an image tag: inspect the final path component.
      case ${image##*/} in
        *:*) repository=${image%:*}; validate_tag "${image##*:}" "$option" ;;
        *) die_usage "$option requires an explicit release tag or SHA-256 digest" ;;
      esac
      ;;
  esac
  [[ "$repository" =~ ^[a-zA-Z0-9][a-zA-Z0-9._:/-]*$ ]] || die_usage "$option has an invalid image repository"
}

PATHS=()
RUN_CONTAINER_CHECK=0
RUN_PARABRICKS_CHECK=0
CUDA_TEST_IMAGE=$DEFAULT_CUDA_TEST_IMAGE
PARABRICKS_VERSION=""
PARABRICKS_IMAGE=""
FORMAT="text"
TIMEOUT=$DEFAULT_TIMEOUT_SECONDS

while [ $# -gt 0 ]; do
  case $1 in
    -h|--help) usage; exit 0 ;;
    --run-container-check) RUN_CONTAINER_CHECK=1; shift ;;
    --run-parabricks-check) RUN_PARABRICKS_CHECK=1; shift ;;
    --path=*) PATHS+=("${1#*=}"); shift ;;
    --cuda-test-image=*) CUDA_TEST_IMAGE=${1#*=}; shift ;;
    --parabricks-version=*) PARABRICKS_VERSION=${1#*=}; shift ;;
    --parabricks-image=*) PARABRICKS_IMAGE=${1#*=}; shift ;;
    --format=*) FORMAT=${1#*=}; shift ;;
    --timeout=*) TIMEOUT=${1#*=}; shift ;;
    --path|--cuda-test-image|--parabricks-version|--parabricks-image|--format|--timeout)
      [ $# -ge 2 ] || die_usage "argument $1: expected one argument"
      case $1 in
        --path) PATHS+=("$2") ;;
        --cuda-test-image) CUDA_TEST_IMAGE=$2 ;;
        --parabricks-version) PARABRICKS_VERSION=$2 ;;
        --parabricks-image) PARABRICKS_IMAGE=$2 ;;
        --format) FORMAT=$2 ;;
        --timeout) TIMEOUT=$2 ;;
      esac
      shift 2
      ;;
    *) die_usage "unrecognized arguments: $1" ;;
  esac
done

case $FORMAT in
  text|json) ;;
  *) die_usage "argument --format: invalid choice: '$FORMAT' (choose from 'text', 'json')" ;;
esac
case $TIMEOUT in
  ''|*[!0-9]*) die_usage "argument --timeout: invalid int value: '$TIMEOUT'" ;;
esac
[ -n "${TIMEOUT//0/}" ] || die_usage "argument --timeout: must be greater than zero"
if [ -n "$PARABRICKS_VERSION" ] && [ -n "$PARABRICKS_IMAGE" ]; then
  die_usage "choose --parabricks-version or --parabricks-image, not both"
fi
if [ "$RUN_PARABRICKS_CHECK" = 1 ] && [ -z "$PARABRICKS_VERSION$PARABRICKS_IMAGE" ]; then
  die_usage "--run-parabricks-check requires --parabricks-version TAG or --parabricks-image IMAGE"
fi
validate_image_reference "$CUDA_TEST_IMAGE" --cuda-test-image
if [ -n "$PARABRICKS_VERSION" ]; then
  validate_tag "$PARABRICKS_VERSION" --parabricks-version
fi
if [ -n "$PARABRICKS_IMAGE" ]; then
  validate_image_reference "$PARABRICKS_IMAGE" --parabricks-image
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_BIN=timeout
elif command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_BIN=gtimeout
else
  printf '%s: error: GNU timeout or gtimeout is required; no probes were run.\n' "$PROG" >&2
  exit 2
fi

TMP_DIR=$(mktemp -d) || exit 1
trap 'rm -rf "$TMP_DIR"' EXIT

trim_str() {
  local s=$1 leading trailing
  leading=${s%%[![:space:]]*}
  s=${s#"$leading"}
  trailing=${s##*[![:space:]]}
  s=${s%"$trailing"}
  printf '%s' "$s"
}

# Sets CMD_RC (empty when the command was not found or timed out), CMD_OUT,
# CMD_ERR, and CMD_TIMED_OUT.
run_cmd() {
  CMD_TIMED_OUT=0
  if ! command -v "$1" >/dev/null 2>&1; then
    CMD_RC=""; CMD_OUT=""; CMD_ERR="command not found"
    return
  fi
  "$TIMEOUT_BIN" --kill-after="$TIMEOUT_KILL_AFTER_SECONDS" "$TIMEOUT" "$@" \
    >"$TMP_DIR/out" 2>"$TMP_DIR/err" </dev/null
  CMD_RC=$?
  if [ "$CMD_RC" -eq 124 ] || [ "$CMD_RC" -eq 137 ]; then
    CMD_TIMED_OUT=1; CMD_RC=""
  fi
  CMD_OUT=$(trim_str "$(cat "$TMP_DIR/out")")
  CMD_ERR=$(trim_str "$(cat "$TMP_DIR/err")")
}

summarize_result() {
  if [ "$CMD_TIMED_OUT" = 1 ]; then
    printf 'command timed out'
    return
  fi
  local output=${CMD_ERR:-${CMD_OUT:-no output}}
  output=${output%%$'\n'*}
  printf '%s' "${output:0:$SUMMARY_MAX_CHARS}"
}

is_number() {
  awk -v v="$1" 'BEGIN { exit !(v ~ /^[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$/) }'
}

lt() {
  awk -v a="$1" -v b="$2" 'BEGIN { exit !(a + 0 < b + 0) }'
}

# Divide by a power of 1024 and round to 2 decimals; prints like Python's {:g}.
scale_round() {
  awk -v v="$1" -v d="$2" 'BEGIN { x = sprintf("%.2f", v / d); print x + 0 }'
}

format_gb() {
  if [ -z "$1" ]; then printf 'unknown'; else printf '%s GB' "$1"; fi
}

# Append a line to a newline-delimited list variable unless already present.
add_unique() {
  local var=$1 value=$2
  case $'\n'"${!var}" in
    *$'\n'"$value"$'\n'*) return 0 ;;
  esac
  printf -v "$var" '%s%s\n' "${!var}" "$value"
}

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

check_target() {
  HOST=$(uname -n 2>/dev/null)
  SYSTEM=$(uname -s)
  RELEASE=$(uname -r)
  MACHINE=$(uname -m)
  PLATFORM="$SYSTEM-$RELEASE-$MACHINE"
}

os_release_value() {
  local value
  value=$(sed -n "s/^$1=//p" /etc/os-release 2>/dev/null | tail -n 1)
  value=$(trim_str "$value")
  value=${value#\"}; value=${value%\"}
  printf '%s' "$value"
}

check_os() {
  OS_STATUS=ok
  OS_DETAIL=""
  OS_DISTRIBUTION=""
  if [ -r /etc/os-release ]; then
    OS_DISTRIBUTION=$(os_release_value PRETTY_NAME)
    [ -n "$OS_DISTRIBUTION" ] || OS_DISTRIBUTION=$(os_release_value NAME)
  fi
  if [ "$SYSTEM" != "Linux" ]; then
    OS_STATUS=warning
    OS_DETAIL="Parabricks container execution is expected on Linux hosts."
  fi
}

check_cpu_memory() {
  CPU_COUNT=$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null)
  case $CPU_COUNT in ''|*[!0-9]*|0) CPU_COUNT="" ;; esac
  MEM_TOTAL_GB=""
  if [ -r /proc/meminfo ]; then
    MEM_TOTAL_GB=$(awk '$1 == "MemTotal:" && $2 ~ /^[0-9]+$/ {
      x = sprintf("%.2f", $2 / 1048576); print x + 0; exit }' /proc/meminfo)
  fi
  CPU_DETAIL=""
  if [ -n "$CPU_COUNT" ]; then CPU_STATUS=ok; else CPU_STATUS=warning; fi
  if [ -z "$MEM_TOTAL_GB" ]; then
    CPU_STATUS=warning
    CPU_DETAIL="Could not determine total system memory from /proc/meminfo."
  fi
}

check_nvidia_smi() {
  GPU_NAMES=(); GPU_TOTAL=(); GPU_FREE=(); GPU_DRIVER=(); GPU_CC=()
  GPU_CUDA=""
  GPU_DETAIL=""
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    GPU_STATUS=missing
    GPU_DETAIL="nvidia-smi was not found on PATH."
    return
  fi

  run_cmd nvidia-smi \
    --query-gpu=name,memory.total,memory.free,driver_version,compute_cap \
    --format=csv,noheader,nounits
  if [ "$CMD_RC" != 0 ]; then
    GPU_STATUS=failed
    GPU_DETAIL="nvidia-smi failed: $(summarize_result)"
    return
  fi
  local query_out=$CMD_OUT

  # CUDA is a driver-level capability, not a supported --query-gpu field.
  # Keep GPU discovery usable even when the optional banner probe fails.
  run_cmd nvidia-smi
  if [ "$CMD_RC" = 0 ]; then
    GPU_CUDA=$(grep -oE 'CUDA Version:[[:space:]]*[0-9]+(\.[0-9]+)+' <<<"$CMD_OUT" \
      | head -n 1 | grep -oE '[0-9]+(\.[0-9]+)+')
  fi

  local line commas f_name f_total f_free f_driver f_cc _rest
  while IFS= read -r line; do
    commas=${line//[^,]/}
    [ ${#commas} -ge 4 ] || continue
    IFS=',' read -r f_name f_total f_free f_driver f_cc _rest <<<"$line"
    f_total=$(trim_str "$f_total")
    f_free=$(trim_str "$f_free")
    GPU_NAMES+=("$(trim_str "$f_name")")
    if is_number "$f_total"; then GPU_TOTAL+=("$(scale_round "$f_total" 1024)"); else GPU_TOTAL+=(""); fi
    if is_number "$f_free"; then GPU_FREE+=("$(scale_round "$f_free" 1024)"); else GPU_FREE+=(""); fi
    GPU_DRIVER+=("$(trim_str "$f_driver")")
    GPU_CC+=("$(trim_str "$f_cc")")
  done <<<"$query_out"

  if [ ${#GPU_NAMES[@]} -gt 0 ]; then
    GPU_STATUS=ok
    if [ -z "$GPU_CUDA" ]; then
      GPU_DETAIL="GPU details are available; the driver-supported CUDA version could not be read."
    fi
  else
    GPU_STATUS=warning
    GPU_DETAIL="nvidia-smi ran, but no GPU rows were parsed."
  fi
}

docker_version_too_old() {
  local major minor rest
  major=${1%%.*}
  rest=${1#*.}
  minor=${rest%%.*}
  [ "$major" -lt "$MIN_DOCKER_MAJOR" ] \
    || { [ "$major" -eq "$MIN_DOCKER_MAJOR" ] && [ "$minor" -lt "$MIN_DOCKER_MINOR" ]; }
}

check_docker() {
  DOCKER_VERSION_OUTPUT=""
  DOCKER_VERSION=""
  DOCKER_DAEMON_ACCESS=""
  DOCKER_DETAIL=""
  if ! command -v docker >/dev/null 2>&1; then
    DOCKER_STATUS=missing
    DOCKER_DETAIL="docker was not found on PATH."
    return
  fi

  run_cmd docker --version
  if [ "$CMD_RC" != 0 ]; then
    DOCKER_STATUS=failed
    DOCKER_DETAIL="docker --version failed: $(summarize_result)"
    return
  fi
  DOCKER_VERSION_OUTPUT=$CMD_OUT
  DOCKER_VERSION=$(grep -oiE 'version[[:space:]]+[0-9]+(\.[0-9]+){1,2}' <<<"$CMD_OUT" \
    | head -n 1 | awk '{ print $2 }')

  run_cmd docker info --format '{{json .}}'
  DOCKER_STATUS=ok
  local details=""
  if [ -n "$DOCKER_VERSION" ] && docker_version_too_old "$DOCKER_VERSION"; then
    DOCKER_STATUS=warning
    details="Docker is older than the documented minimum $MIN_DOCKER_MAJOR.$MIN_DOCKER_MINOR."
  fi
  if [ "$CMD_RC" = 0 ]; then
    DOCKER_DAEMON_ACCESS=true
  else
    DOCKER_DAEMON_ACCESS=false
    DOCKER_STATUS=warning
    details="${details:+$details }docker info failed: $(summarize_result). Docker daemon may be unavailable or permission-restricted."
  fi
  DOCKER_DETAIL=$details
}

check_docker_gpu_container() {
  DGC_IMAGE=""
  DGC_DETAIL=""
  DGC_REASON=""
  if [ "$RUN_CONTAINER_CHECK" != 1 ]; then
    DGC_STATUS=not_checked
    DGC_REASON="Use --run-container-check to test Docker GPU access."
    return
  fi
  if [ "$DOCKER_STATUS" = missing ]; then
    DGC_STATUS=missing
    DGC_DETAIL="Docker is missing, so Docker GPU access cannot be tested."
    return
  fi
  DGC_IMAGE=$CUDA_TEST_IMAGE
  run_cmd docker run --rm --gpus all "$CUDA_TEST_IMAGE" nvidia-smi
  if [ "$CMD_RC" = 0 ]; then
    DGC_STATUS=ok
    DGC_DETAIL="Docker can run nvidia-smi with GPUs."
  else
    DGC_STATUS=failed
    DGC_DETAIL=$(summarize_result)
  fi
}

check_parabricks_container() {
  PB_IMAGE=""
  PB_DETAIL=""
  PB_REASON=""
  if [ "$RUN_PARABRICKS_CHECK" != 1 ]; then
    PB_STATUS=not_checked
    PB_REASON="Use --run-parabricks-check with --parabricks-version <tag> or --parabricks-image <image> to test pbrun."
    return
  fi
  if [ "$DOCKER_STATUS" = missing ]; then
    PB_STATUS=missing
    PB_DETAIL="Docker is missing, so Parabricks container access cannot be tested."
    return
  fi
  PB_IMAGE=${PARABRICKS_IMAGE:-"nvcr.io/nvidia/clara/clara-parabricks:$PARABRICKS_VERSION"}
  run_cmd docker run --rm --gpus all "$PB_IMAGE" pbrun --help
  if [ "$CMD_RC" = 0 ]; then
    PB_STATUS=ok
    PB_DETAIL="Parabricks container started and pbrun responded."
  else
    PB_STATUS=failed
    PB_DETAIL=$(summarize_result)
  fi
}

ST_PATH=(); ST_CHECKED=(); ST_STATUS=(); ST_TOTAL=(); ST_FREE=(); ST_USED=(); ST_DETAIL=()

check_storage_path() {
  local path=$1 target parent total_kib used_kib free_kib
  case $path in
    "~") path=$HOME ;;
    "~/"*) path="$HOME/${path#"~/"}" ;;
  esac

  target=$path
  while [ ! -e "$target" ]; do
    parent=$(dirname -- "$target")
    if [ "$parent" = "$target" ]; then
      target=""
      break
    fi
    target=$parent
  done

  ST_PATH+=("$path")
  if [ -z "$target" ]; then
    ST_CHECKED+=(""); ST_STATUS+=(missing); ST_TOTAL+=(""); ST_FREE+=(""); ST_USED+=("")
    ST_DETAIL+=("Path and all parents are missing.")
    return
  fi

  run_cmd df -Pk -- "$target"
  if [ "$CMD_RC" = 0 ]; then
    read -r total_kib used_kib free_kib <<<"$(awk 'NR == 2 { print $2, $3, $4 }' <<<"$CMD_OUT")"
  fi
  if [ "$CMD_RC" != 0 ] || [ -z "$total_kib" ]; then
    ST_CHECKED+=("$target"); ST_STATUS+=(failed); ST_TOTAL+=(""); ST_FREE+=(""); ST_USED+=("")
    ST_DETAIL+=("df failed for $target: $(summarize_result)")
    return
  fi

  ST_CHECKED+=("$target")
  if [ "$free_kib" -gt 0 ]; then ST_STATUS+=(ok); else ST_STATUS+=(warning); fi
  ST_TOTAL+=("$(scale_round "$total_kib" 1048576)")
  ST_FREE+=("$(scale_round "$free_kib" 1048576)")
  ST_USED+=("$(scale_round "$used_kib" 1048576)")
  ST_DETAIL+=("")
}

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

classify() {
  ASSESSMENTS=""
  RECOMMENDATIONS=""
  OPEN_QUESTIONS=""
  local i low_memory=0 low_storage=0 missing_storage=0

  if [ "$OS_STATUS" = warning ]; then
    add_unique ASSESSMENTS "not supported"
    add_unique RECOMMENDATIONS "Run Parabricks on a Linux host supported by Docker and NVIDIA Container Toolkit."
  fi

  if [ "$GPU_STATUS" = missing ]; then
    add_unique ASSESSMENTS "runtime-incomplete"
    add_unique RECOMMENDATIONS "Make NVIDIA drivers and nvidia-smi available on the target execution host."
  elif [ "$GPU_STATUS" != ok ]; then
    add_unique ASSESSMENTS "runtime-incomplete"
    add_unique RECOMMENDATIONS "Resolve nvidia-smi errors before running Parabricks."
  else
    for i in "${!GPU_NAMES[@]}"; do
      if [ -n "${GPU_TOTAL[$i]}" ] && lt "${GPU_TOTAL[$i]}" "$MIN_GPU_MEMORY_GB"; then
        low_memory=1
      fi
    done
    if [ "$low_memory" = 1 ]; then
      add_unique ASSESSMENTS "hardware-constrained"
      add_unique RECOMMENDATIONS "Use a GPU with at least $MIN_GPU_MEMORY_GB GB memory per GPU, or select workflows/options documented for lower memory."
    fi
  fi

  if [ "$DOCKER_STATUS" = missing ]; then
    add_unique ASSESSMENTS "runtime-incomplete"
    add_unique RECOMMENDATIONS "Install and configure Docker Engine on the target host."
  elif [ "$DOCKER_STATUS" != ok ]; then
    add_unique ASSESSMENTS "runtime-incomplete"
    add_unique RECOMMENDATIONS "Resolve Docker daemon/version/access issues before running Parabricks."
  fi

  if [ "$DGC_STATUS" = not_checked ]; then
    add_unique ASSESSMENTS "not enough information"
    add_unique OPEN_QUESTIONS "Docker GPU container access was not tested; rerun with --run-container-check on the target host."
  elif [ "$DGC_STATUS" != ok ]; then
    add_unique ASSESSMENTS "runtime-incomplete"
    add_unique RECOMMENDATIONS "Configure NVIDIA Container Toolkit so Docker can run containers with --gpus all."
  fi

  if [ "$PB_STATUS" = not_checked ]; then
    add_unique ASSESSMENTS "not enough information"
    add_unique OPEN_QUESTIONS "Parabricks container access was not tested; use --run-parabricks-check with --parabricks-version <tag> or --parabricks-image <image> after selecting an image and authorizing the probe."
  elif [ "$PB_STATUS" != ok ]; then
    add_unique ASSESSMENTS "runtime-incomplete"
    add_unique RECOMMENDATIONS "Verify NGC authentication, network/proxy policy, and the requested Parabricks image tag."
  fi

  if [ -z "$CPU_COUNT" ] || [ -z "$MEM_TOTAL_GB" ]; then
    add_unique ASSESSMENTS "not enough information"
    add_unique OPEN_QUESTIONS "CPU count or total system memory could not be determined."
  fi

  for i in "${!ST_PATH[@]}"; do
    if [ -n "${ST_FREE[$i]}" ] && lt "${ST_FREE[$i]}" "$LOW_STORAGE_WARNING_GB"; then
      low_storage=1
    fi
    [ "${ST_STATUS[$i]}" = missing ] && missing_storage=1
    if [ "${ST_STATUS[$i]}" = failed ]; then
      add_unique ASSESSMENTS "not enough information"
      add_unique OPEN_QUESTIONS "Storage probe failed for ${ST_PATH[$i]}: ${ST_DETAIL[$i]}"
    fi
  done
  if [ "$low_storage" = 1 ]; then
    add_unique ASSESSMENTS "I/O-constrained"
    add_unique RECOMMENDATIONS "Check input, output, and temporary filesystems; less than $LOW_STORAGE_WARNING_GB GB free may be risky for many workflows."
  fi
  if [ "$missing_storage" = 1 ]; then
    add_unique ASSESSMENTS "not enough information"
    add_unique OPEN_QUESTIONS "One or more requested storage paths do not exist on this host."
  fi

  if [ -z "$ASSESSMENTS" ]; then
    add_unique ASSESSMENTS "ready"
    add_unique RECOMMENDATIONS "Required local runtime checks passed. Confirm workflow-specific storage, reference files, and Parabricks image access before production runs."
  fi
}

# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

render_text() {
  local i line joined=""
  echo "Environment summary:"
  echo "- Host: ${HOST:-unknown}"
  echo "- Platform: $PLATFORM ($MACHINE)"
  echo "- OS: ${OS_DISTRIBUTION:-$SYSTEM} $RELEASE"
  echo "- CPU/RAM: ${CPU_COUNT:-unknown} CPUs, $(format_gb "$MEM_TOTAL_GB") RAM"
  echo "- Shell: bash $BASH_VERSION ($BASH)"

  if [ "$GPU_STATUS" = ok ]; then
    echo "- GPUs: ${#GPU_NAMES[@]}"
    for i in "${!GPU_NAMES[@]}"; do
      echo "  - GPU $((i + 1)): ${GPU_NAMES[$i]}, $(format_gb "${GPU_TOTAL[$i]}") total, driver ${GPU_DRIVER[$i]}, driver-supported CUDA ${GPU_CUDA:-unknown}, CC ${GPU_CC[$i]}"
    done
  else
    echo "- GPUs: $GPU_STATUS (${GPU_DETAIL:-not available})"
  fi

  echo "- Docker: $DOCKER_STATUS${DOCKER_VERSION_OUTPUT:+, $DOCKER_VERSION_OUTPUT}"
  echo "- Docker GPU container: $DGC_STATUS"
  echo "- Parabricks container: $PB_STATUS"

  if [ ${#ST_PATH[@]} -gt 0 ]; then
    echo "- Storage:"
    for i in "${!ST_PATH[@]}"; do
      if [ "${ST_STATUS[$i]}" = ok ]; then
        echo "  - ${ST_PATH[$i]}: $(format_gb "${ST_FREE[$i]}") free of $(format_gb "${ST_TOTAL[$i]}") (checked ${ST_CHECKED[$i]})"
      else
        echo "  - ${ST_PATH[$i]}: ${ST_STATUS[$i]} (${ST_DETAIL[$i]:-not available})"
      fi
    done
  fi

  echo
  echo "Assessment:"
  while IFS= read -r line; do
    [ -n "$line" ] && joined="${joined:+$joined, }$line"
  done <<<"$ASSESSMENTS"
  echo "$joined"

  echo
  echo "Recommendations:"
  while IFS= read -r line; do
    [ -n "$line" ] && echo "- $line"
  done <<<"$RECOMMENDATIONS"

  echo
  echo "Open questions:"
  if [ -n "$OPEN_QUESTIONS" ]; then
    while IFS= read -r line; do
      [ -n "$line" ] && echo "- $line"
    done <<<"$OPEN_QUESTIONS"
  else
    echo "- None from this diagnostic run."
  fi
}

# ---------------------------------------------------------------------------
# JSON rendering (sorted keys, 2-space indent)
# ---------------------------------------------------------------------------

json_str() {
  local s
  s=$(tr -d '\000-\010\013\014\016-\037' <<<"$1")
  s=${s//\\/\\\\}
  s=${s//\"/\\\"}
  s=${s//$'\n'/\\n}
  s=${s//$'\r'/\\r}
  s=${s//$'\t'/\\t}
  printf '"%s"' "$s"
}

json_str_or_null() {
  if [ -z "$1" ]; then printf 'null'; else json_str "$1"; fi
}

# Empty output means "omit this key" in json_obj.
json_opt_str() {
  [ -n "$1" ] && json_str "$1"
}

json_num() {
  if [ -z "$1" ]; then printf 'null'; else printf '%s' "$1"; fi
}

# json_obj INDENT key rendered_value [key rendered_value ...]
# Pairs whose rendered value is empty are omitted.
json_obj() {
  local indent=$1 pad inner key sep="" out="{"
  shift
  printf -v pad '%*s' "$indent" ''
  inner="$pad  "
  while [ $# -ge 2 ]; do
    if [ -n "$2" ]; then
      key=$(json_str "$1")
      out+="$sep"
      out+=$'\n'
      out+="$inner$key: $2"
      sep=","
    fi
    shift 2
  done
  if [ -n "$sep" ]; then
    out+=$'\n'
    out+="$pad"
  fi
  printf '%s}' "$out"
}

# json_arr INDENT rendered_item [rendered_item ...]
json_arr() {
  local indent=$1 pad inner item sep="" out="["
  shift
  printf -v pad '%*s' "$indent" ''
  inner="$pad  "
  for item in "$@"; do
    out+="$sep"
    out+=$'\n'
    out+="$inner$item"
    sep=","
  done
  if [ -n "$sep" ]; then
    out+=$'\n'
    out+="$pad"
  fi
  printf '%s]' "$out"
}

lines_to_json_arr() {
  local indent=$1 line items=()
  while IFS= read -r line; do
    [ -n "$line" ] && items+=("$(json_str "$line")")
  done <<<"$2"
  json_arr "$indent" "${items[@]}"
}

render_json() {
  local i cpu docker dgc nvsmi os pb shell checks target gpus_json=""
  local gpus=() storage=()

  cpu=$(json_obj 4 \
    cpu_count "$(json_num "$CPU_COUNT")" \
    detail "$(json_opt_str "$CPU_DETAIL")" \
    memory_total_gb "$(json_num "$MEM_TOTAL_GB")" \
    status "$(json_str "$CPU_STATUS")")

  if [ "$DOCKER_STATUS" = missing ] || [ "$DOCKER_STATUS" = failed ]; then
    docker=$(json_obj 4 \
      detail "$(json_str "$DOCKER_DETAIL")" \
      status "$(json_str "$DOCKER_STATUS")")
  else
    docker=$(json_obj 4 \
      daemon_access "$DOCKER_DAEMON_ACCESS" \
      detail "$(json_str_or_null "$DOCKER_DETAIL")" \
      status "$(json_str "$DOCKER_STATUS")" \
      version "$(json_str_or_null "$DOCKER_VERSION")" \
      version_output "$(json_str "$DOCKER_VERSION_OUTPUT")")
  fi

  if [ "$DGC_STATUS" = not_checked ]; then
    dgc=$(json_obj 4 reason "$(json_str "$DGC_REASON")" status "$(json_str "$DGC_STATUS")")
  else
    dgc=$(json_obj 4 \
      detail "$(json_str "$DGC_DETAIL")" \
      image "$(json_opt_str "$DGC_IMAGE")" \
      status "$(json_str "$DGC_STATUS")")
  fi

  if [ "$GPU_STATUS" != failed ]; then
    for i in "${!GPU_NAMES[@]}"; do
      gpus+=("$(json_obj 8 \
        compute_capability "$(json_str "${GPU_CC[$i]}")" \
        cuda_version "$(json_str_or_null "$GPU_CUDA")" \
        driver_version "$(json_str "${GPU_DRIVER[$i]}")" \
        memory_free_gb "$(json_num "${GPU_FREE[$i]}")" \
        memory_total_gb "$(json_num "${GPU_TOTAL[$i]}")" \
        name "$(json_str "${GPU_NAMES[$i]}")")")
    done
    gpus_json=$(json_arr 6 "${gpus[@]}")
  fi
  nvsmi=$(json_obj 4 \
    detail "$(json_opt_str "$GPU_DETAIL")" \
    gpus "$gpus_json" \
    status "$(json_str "$GPU_STATUS")")

  os=$(json_obj 4 \
    detail "$(json_opt_str "$OS_DETAIL")" \
    distribution "$(json_str_or_null "$OS_DISTRIBUTION")" \
    release "$(json_str "$RELEASE")" \
    status "$(json_str "$OS_STATUS")" \
    system "$(json_str "$SYSTEM")")

  if [ "$PB_STATUS" = not_checked ]; then
    pb=$(json_obj 4 reason "$(json_str "$PB_REASON")" status "$(json_str "$PB_STATUS")")
  else
    pb=$(json_obj 4 \
      detail "$(json_str "$PB_DETAIL")" \
      image "$(json_str "$PB_IMAGE")" \
      status "$(json_str "$PB_STATUS")")
  fi

  shell=$(json_obj 4 \
    executable "$(json_str "$BASH")" \
    status '"ok"' \
    version "$(json_str "$BASH_VERSION")")

  checks=$(json_obj 2 \
    cpu_memory "$cpu" \
    docker "$docker" \
    docker_gpu_container "$dgc" \
    nvidia_smi "$nvsmi" \
    os "$os" \
    parabricks_container "$pb" \
    shell "$shell")

  for i in "${!ST_PATH[@]}"; do
    if [ "${ST_STATUS[$i]}" = ok ] || [ "${ST_STATUS[$i]}" = warning ]; then
      storage+=("$(json_obj 4 \
        checked_path "$(json_str "${ST_CHECKED[$i]}")" \
        free_gb "$(json_num "${ST_FREE[$i]}")" \
        path "$(json_str "${ST_PATH[$i]}")" \
        status "$(json_str "${ST_STATUS[$i]}")" \
        total_gb "$(json_num "${ST_TOTAL[$i]}")" \
        used_gb "$(json_num "${ST_USED[$i]}")")")
    else
      storage+=("$(json_obj 4 \
        detail "$(json_str "${ST_DETAIL[$i]}")" \
        path "$(json_str "${ST_PATH[$i]}")" \
        status "$(json_str "${ST_STATUS[$i]}")")")
    fi
  done

  target=$(json_obj 2 \
    host "$(json_str_or_null "$HOST")" \
    machine "$(json_str "$MACHINE")" \
    platform "$(json_str "$PLATFORM")" \
    shell "$(json_str "$BASH_VERSION")")

  json_obj 0 \
    assessment "$(lines_to_json_arr 2 "$ASSESSMENTS")" \
    checks "$checks" \
    open_questions "$(lines_to_json_arr 2 "$OPEN_QUESTIONS")" \
    recommendations "$(lines_to_json_arr 2 "$RECOMMENDATIONS")" \
    schema_version "$SCHEMA_VERSION" \
    storage "$(json_arr 2 "${storage[@]}")" \
    target "$target"
  echo
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

check_target
check_os
check_cpu_memory
check_nvidia_smi
check_docker
check_docker_gpu_container
check_parabricks_container
for storage_path in "${PATHS[@]}"; do
  check_storage_path "$storage_path"
done
classify

if [ "$FORMAT" = json ]; then
  render_json
else
  render_text
fi
exit 0
