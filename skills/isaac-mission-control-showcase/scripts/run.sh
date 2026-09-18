#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHOWCASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

resolve_skills_dir() {
  local candidate="$SHOWCASE_DIR"
  while [[ "$candidate" != "/" ]]; do
    if [[ "$(basename "$candidate")" == "skills" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
    candidate="$(dirname "$candidate")"
  done
  echo "ERROR: showcase skill is not installed beneath a skills directory" >&2
  return 1
}

# Upstream skills are owned by their own repositories and resolved at runtime.
# The runner drives them through the parent-owned reference adapters under
# references/<component>/scripts/run.py, never by package-relative vendored
# paths and never by a hardcoded upstream path. Nothing here clones, fetches,
# pulls, installs, or updates a dependency.
REFERENCE_DIR="$SHOWCASE_DIR/references"
DOCTOR="${DOCTOR:-$SCRIPT_DIR/doctor.py}"

adapter() {
  printf '%s\n' "$REFERENCE_DIR/$1/scripts/run.py"
}

# Resolved upstream skill directory, read from the persisted manifest. Relative
# resources belong to the upstream skill, not to this package.
component_dir() {
  SHOWCASE_SHARED="$SHOWCASE_DIR/shared" python3 -c '
import os, sys
sys.path.insert(0, os.environ["SHOWCASE_SHARED"])
import dependency_manifest as dm
try:
    print(dm.skill_dir(dm.load(), sys.argv[1]))
except dm.Blocked as exc:
    print(f"BLOCKED: {exc}", file=sys.stderr); raise SystemExit(3)
' "$1"
}

# Resolved upstream entrypoint for tools that inspect resources relative to the
# upstream skill. Runtime invocation still goes through the reference adapter.
component_entrypoint() {
  SHOWCASE_SHARED="$SHOWCASE_DIR/shared" python3 -c '
import os, sys
sys.path.insert(0, os.environ["SHOWCASE_SHARED"])
import dependency_manifest as dm
try:
    print(dm.entrypoint(dm.load(), sys.argv[1], sys.argv[2]))
except dm.Blocked as exc:
    print(f"BLOCKED: {exc}", file=sys.stderr); raise SystemExit(3)
' "$1" "$2"
}

verify_dependencies() {
  # Read-only freshness check. The adapters consume the persisted manifest.
  if ! python3 "$DOCTOR" dependencies --check-only >/dev/null 2>&1; then
    python3 "$DOCTOR" dependencies --check-only >&2 || true
    echo "ERROR: showcase dependencies are not ready." >&2
    echo "       Resolve them once with:" >&2
    echo "         python3 $DOCTOR dependencies --prepare" >&2
    return 1
  fi
}

SKILLS_DIR="$(resolve_skills_dir)"
# The adapters are the only path to an upstream component. The legacy
# per-script overrides pointed at an upstream entrypoint directly, which
# bypassed manifest verification, the readiness gate, and the upstream working
# directory. They are refused rather than ignored, so a stale export fails
# loudly instead of silently changing which code runs.
reject_legacy_overrides() {
  local var legacy=(BRING_UP_CLOUD CHANGE_MAP CHANGE_FLEET ISAAC_SEND)
  local found=()
  for var in "${legacy[@]}"; do
    [[ -n "${!var:-}" ]] && found+=("$var")
  done
  if (( ${#found[@]} )); then
    echo "ERROR: these overrides bypass the reference adapters and are no longer" >&2
    echo "       supported: ${found[*]}" >&2
    echo "       They pointed directly at an upstream entrypoint, skipping" >&2
    echo "       manifest verification and the readiness gate." >&2
    echo "       Unset them. To use a different checkout, point the component at" >&2
    echo "       it and re-run preflight, for example:" >&2
    echo "         export CHANGE_MAP_SKILL_DIR=/path/to/mission-control/skills/change-map" >&2
    echo "         python3 $DOCTOR dependencies --check-only" >&2
    return 2
  fi
}

# Reject a bypassing override before anything else: it is a configuration
# error regardless of whether dependencies happen to resolve.
reject_legacy_overrides

RESTORE_ISAAC="${RESTORE_ISAAC:-$SCRIPT_DIR/restore_isaac_scene.py}"
SHOWCASE_HELPER="${SHOWCASE_HELPER:-$SCRIPT_DIR/showcase.py}"

# Dependency resolution is deferred until a command needs it. Resolving at load
# time made every invocation, including the --stop recovery path, depend on the
# upstream checkouts being present and pinned. A run that has containers to
# clean up must still be able to clean them up after its dependencies moved.
CLOUD_RESOURCES_RESOLVED=0
RUNTIME_ENTRYPOINTS_RESOLVED=0
CLOUD_UPSTREAM_ENTRYPOINT=""
CLOUD_RESOURCE_DIR=""
CLOUD_COMPOSE_FILE=""
CLOUD_ENV_FILE=""
CLOUD_IMAGES_FILE=""
CLOUD_INIT_DB=""

# Cloud file paths only. --stop needs these to read the compose defaults and
# export INIT_DB_SH; it needs none of the four adapters.
resolve_cloud_resources() {
  (( CLOUD_RESOURCES_RESOLVED )) && return 0
  local resource_dir
  resource_dir="$(component_dir bring-up-cloud-stack)" || return 1
  CLOUD_RESOURCE_DIR="$resource_dir/resources"
  CLOUD_COMPOSE_FILE="$CLOUD_RESOURCE_DIR/docker-compose/bringup_services.yaml"
  CLOUD_ENV_FILE="$CLOUD_RESOURCE_DIR/docker-compose/.env"
  CLOUD_IMAGES_FILE="$CLOUD_RESOURCE_DIR/docker-compose/images.conf"
  CLOUD_INIT_DB="$CLOUD_RESOURCE_DIR/docker-compose/init-db.sh"
  CLOUD_RESOURCES_RESOLVED=1
}

# Everything a run needs to start components. Fatal on any path that launches.
resolve_runtime_entrypoints() {
  (( RUNTIME_ENTRYPOINTS_RESOLVED )) && return 0
  verify_dependencies || return 1
  resolve_cloud_resources || return 1
  CLOUD_UPSTREAM_ENTRYPOINT="$(component_entrypoint bring-up-cloud-stack up)" || return 1
  BRING_UP_CLOUD="$(adapter bring-up-cloud-stack)" || return 1
  CHANGE_MAP="$(adapter change-map)" || return 1
  CHANGE_FLEET="$(adapter change-fleet-composition)" || return 1
  ISAAC_SEND="$(adapter isaac-sim-remote)" || return 1
  require_file "$BRING_UP_CLOUD"
  require_file "$CHANGE_MAP"
  require_file "$CHANGE_FLEET"
  require_file "$ISAAC_SEND"
  require_file "$CLOUD_COMPOSE_FILE"
  RUNTIME_ENTRYPOINTS_RESOLVED=1
}

LKG_WAREHOUSE_USD_URI="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1/Isaac/Environments/Simple_Warehouse/warehouse.usd"
MAP_BASENAME="carter_warehouse_navigation"
MAP_PNG="${MAP_BASENAME}.png"
MAP_YAML="${MAP_BASENAME}.yaml"
MAP_ID="$MAP_PNG"
EXPECTED_MAP_SHA256="dd2f5e382a5f331866becaeaffb391a7e46b595873bf25c9cbb4e280ec261b8e"

WAREHOUSE_USD_URI_WAS_SET=0
if [[ -n "${WAREHOUSE_USD_URI:-}" ]]; then
  WAREHOUSE_USD_URI_WAS_SET=1
fi

ISAAC_SIM_DIR="${ISAAC_SIM_DIR:-$HOME/isaacsim}"
WAREHOUSE_USD_URI="${WAREHOUSE_USD_URI:-$LKG_WAREHOUSE_USD_URI}"
NOVA_CARTER_IMAGE="${NOVA_CARTER_IMAGE:-nvcr.io/nvidia/isaac/nova_carter_sil:release-3.2}"
NOVA_CARTER_MEMORY_LIMIT="${NOVA_CARTER_MEMORY_LIMIT:-20g}"
NOVA_CARTER_MEMORY_SWAP_LIMIT="${NOVA_CARTER_MEMORY_SWAP_LIMIT:-24g}"
ROBOT_NAME="${ROBOT_NAME:-carter01}"
SHOWCASE_GPU_DEVICE="${SHOWCASE_GPU_DEVICE:-0}"
SHOWCASE_ROS_DOMAIN_ID="${SHOWCASE_ROS_DOMAIN_ID:-77}"
SHOWCASE_ROS_LOCALHOST_ONLY="${SHOWCASE_ROS_LOCALHOST_ONLY:-0}"

ISAAC_PYTHON_PORT="${ISAAC_PYTHON_PORT:-8226}"
WPG_PORT="${WPG_PORT:-8000}"

DRY_RUN=0
STOP=0
DEMO=0
PREFLIGHT=0
PULL_IMAGES=1
SHOWCASE_WORK_DIR="${SHOWCASE_WORK_DIR:-}"
DEMO_MISSION_ID=""
IGNORED_CONTAINERS=()

# Print command usage.
usage() {
  cat <<USAGE
Usage: $0 [--preflight|--demo|--stop] [--robot-name NAME] [--work-dir PATH] [--ignore-container NAME] [--dry-run]

Run the canonical small warehouse + Nova Carter SIL + Mission Control
showcase without requiring Mission Control or Isaac ROS source. Isaac Sim is
launched from a local installation with its GUI window and driven through the
isaac-sim-remote Python server; there is no WebRTC or browser viewer.

Modes:
  no flag             Start and validate the stack, leaving the robot idle.
  --preflight         Run read-only host/dependency checks and exit.
  --demo, --replay    Start, run the deterministic circular route, and verify it.
  --stop              Stop only the last recorded showcase resources.

Options:
  --robot-name NAME   Mission Control and Mission Client identity (default: carter01).
  --work-dir PATH     Use a new or empty runtime directory instead of mktemp under /tmp.
  --ignore-container NAME
                      Allow a running container by exact name; repeat as needed.
  --no-pull           Reuse locally cached images instead of refreshing registry tags.
  --dry-run           Print the resolved commands without changing runtime state.
  -h, --help          Show this message.

Environment overrides:
  ISAAC_SIM_DIR, WAREHOUSE_USD_URI, NOVA_CARTER_IMAGE,
  NOVA_CARTER_MEMORY_LIMIT, NOVA_CARTER_MEMORY_SWAP_LIMIT, ROBOT_NAME,
  SHOWCASE_GPU_DEVICE, SHOWCASE_ROS_DOMAIN_ID,
  SHOWCASE_ROS_LOCALHOST_ONLY, ISAAC_PYTHON_PORT, SHOWCASE_WORK_DIR.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preflight) PREFLIGHT=1 ;;
    --demo|--replay) DEMO=1 ;;
    --stop) STOP=1 ;;
    --robot-name)
      if [[ -z "${2-}" || "${2-}" == -* ]]; then
        echo "ERROR: --robot-name requires a value" >&2
        exit 2
      fi
      ROBOT_NAME="${2:-}"
      shift
      ;;
    --work-dir)
      if [[ -z "${2-}" || "${2-}" == -* ]]; then
        echo "ERROR: --work-dir requires a path" >&2
        exit 2
      fi
      SHOWCASE_WORK_DIR="${2:-}"
      shift
      ;;
    --ignore-container)
      if [[ -z "${2-}" || "${2-}" == -* ]]; then
        echo "ERROR: --ignore-container requires a container name" >&2
        exit 2
      fi
      IGNORED_CONTAINERS+=("${2:-}")
      shift
      ;;
    --no-pull) PULL_IMAGES=0 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if (( PREFLIGHT + STOP + DEMO > 1 )); then
  echo "ERROR: choose only one of --preflight, --demo, or --stop" >&2
  exit 2
fi
if [[ ! "$ROBOT_NAME" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "ERROR: robot name must contain only letters, digits, underscores, or dashes" >&2
  exit 2
fi
for ignored_container in "${IGNORED_CONTAINERS[@]}"; do
  if [[ ! "$ignored_container" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "ERROR: invalid ignored container name: $ignored_container" >&2
    exit 2
  fi
done
# The local installation carries no inspectable version tag, so the caller owns
# the asset contract. Never infer a URI by substituting a version number.
if [[ "$WAREHOUSE_USD_URI_WAS_SET" -eq 0 && ! -d "$ISAAC_SIM_DIR" ]]; then
  echo "ERROR: ISAAC_SIM_DIR does not exist: $ISAAC_SIM_DIR" >&2
  echo "       Set ISAAC_SIM_DIR to a local Isaac Sim 6.1.0 installation, or set" >&2
  echo "       WAREHOUSE_USD_URI explicitly for a different version." >&2
  exit 2
fi

print_command() {
  local argument printable
  printf '+'
  for argument in "$@"; do
    printable="$argument"
    if [[ -n "${MQTT_PASSWORD:-}" ]]; then
      printable="${printable//"$MQTT_PASSWORD"/<redacted>}"
    fi
    printf ' %q' "$printable"
  done
  printf '\n'
}

# Print and run a command, unless this is a dry run.
run() {
  print_command "$@"
  if [[ $DRY_RUN -eq 0 ]]; then
    "$@"
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: required packaged dependency is missing: $path" >&2
    if [[ "$path" != "$SHOWCASE_DIR"/* ]]; then
      echo "       This belongs to an upstream skill resolved at runtime." >&2
      echo "       Re-resolve dependencies with:" >&2
      echo "         python3 $DOCTOR dependencies --check-only" >&2
    fi
    return 1
  fi
}

load_cloud_defaults() {
  require_file "$CLOUD_ENV_FILE"
  require_file "$CLOUD_IMAGES_FILE"
  set -a
  # shellcheck disable=SC1090
  . "$CLOUD_ENV_FILE"
  # shellcheck disable=SC1090
  . "$CLOUD_IMAGES_FILE"
  set +a
  MQTT_REQUIRED_PORT="$MQTT_PORT_WEBSOCKET"
}

current_pointer() {
  printf '%s/mission-control-showcase-current-%s\n' "${TMPDIR:-/tmp}" "$(id -u)"
}

initialize_work_dir() {
  local base run_slug
  if [[ $DRY_RUN -eq 1 ]]; then
    SHOWCASE_WORK_DIR="${SHOWCASE_WORK_DIR:-${TMPDIR:-/tmp}/mission-control-showcase.DRYRUN}"
  elif [[ -n "$SHOWCASE_WORK_DIR" ]]; then
    if [[ -e "$SHOWCASE_WORK_DIR" && ! -d "$SHOWCASE_WORK_DIR" ]]; then
      echo "ERROR: --work-dir is not a directory: $SHOWCASE_WORK_DIR" >&2
      return 1
    fi
    if [[ -d "$SHOWCASE_WORK_DIR" && -n "$(find "$SHOWCASE_WORK_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "ERROR: --work-dir must be new or empty: $SHOWCASE_WORK_DIR" >&2
      return 1
    fi
    run mkdir -p "$SHOWCASE_WORK_DIR"
  else
    SHOWCASE_WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mission-control-showcase.XXXXXX")"
  fi

  SHOWCASE_WORK_DIR="$(cd "$(dirname "$SHOWCASE_WORK_DIR")" && pwd)/$(basename "$SHOWCASE_WORK_DIR")"
  base="$(basename "$SHOWCASE_WORK_DIR")"
  run_slug="$(printf '%s' "$base" | tr '[:upper:].' '[:lower:]-' | tr -cd 'a-z0-9_-')"
  COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-$run_slug}"
  if [[ ! "$COMPOSE_PROJECT_NAME" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
    echo "ERROR: invalid COMPOSE_PROJECT_NAME: $COMPOSE_PROJECT_NAME" >&2
    return 1
  fi

  MC_WORKSPACE="$SHOWCASE_WORK_DIR/mission-control"
  RUNTIME_MAP_DIR="$SHOWCASE_WORK_DIR/runtime-maps"
  ISAAC_DATA_ROOT="$SHOWCASE_WORK_DIR/isaac-sim"
  NOVA_CONTAINER="${COMPOSE_PROJECT_NAME}-nova-carter-sil"
  ISAAC_PID_FILE="$SHOWCASE_WORK_DIR/isaac-sim.pid"
  ISAAC_LOG_FILE="$ISAAC_DATA_ROOT/logs/isaac-sim.log"
  NOVA_LOG_FILE="$SHOWCASE_WORK_DIR/logs/nova-carter-ros2.log"
  STATE_FILE="$SHOWCASE_WORK_DIR/state.env"

  # The map is generated from the live stage before prepare_runtime runs, so
  # these roots must exist ahead of the Isaac Sim launch.
  run mkdir -p "$RUNTIME_MAP_DIR" "$ISAAC_DATA_ROOT/logs" "$SHOWCASE_WORK_DIR/logs"

  if [[ $DRY_RUN -eq 0 ]]; then
    chmod 0700 "$SHOWCASE_WORK_DIR"
    umask 077
    {
      printf 'SHOWCASE_WORK_DIR=%q\n' "$SHOWCASE_WORK_DIR"
      printf 'MC_WORKSPACE=%q\n' "$MC_WORKSPACE"
      printf 'COMPOSE_PROJECT_NAME=%q\n' "$COMPOSE_PROJECT_NAME"
      printf 'NOVA_CONTAINER=%q\n' "$NOVA_CONTAINER"
      printf 'ISAAC_PID_FILE=%q\n' "$ISAAC_PID_FILE"
      printf 'NOVA_LOG_FILE=%q\n' "$NOVA_LOG_FILE"
    } > "$STATE_FILE"
    printf '%s\n' "$SHOWCASE_WORK_DIR" > "$(current_pointer)"
    umask 022
    manifest_init
    trap 'on_interrupt' INT TERM
  fi
  echo "showcase_work_dir=$SHOWCASE_WORK_DIR"
}

manifest_py() {
  python3 -c '
import sys
sys.path.insert(0, sys.argv[1])
import run_state
run_state.__dict__[sys.argv[2]](*sys.argv[3:])
' "$SCRIPT_DIR" "$@"
}

manifest_init() {
  python3 - "$SCRIPT_DIR" "$SHOWCASE_WORK_DIR" "$COMPOSE_PROJECT_NAME" \
    "$NOVA_CONTAINER" "$ROBOT_NAME" "$ISAAC_PYTHON_PORT" \
    "$SHOWCASE_ROS_DOMAIN_ID" "$NOVA_LOG_FILE" <<'PY' || true
import sys
sys.path.insert(0, sys.argv[1])
import run_state
work, project, nova, robot, port, domain, log = sys.argv[2:9]
run_state.initialize(
    work,
    compose_project=project,
    robot=robot,
    ros_domain_id=domain,
    ports={"isaac_python_port": port},
    evidence={"ros2_log": log},
)
run_state.add_container(work, nova)
PY
}

manifest_state() {
  [[ $DRY_RUN -eq 1 || -z "${SHOWCASE_WORK_DIR:-}" ]] && return 0
  manifest_py set_state "$SHOWCASE_WORK_DIR" "$1" 2>/dev/null || true
}

on_interrupt() {
  trap - INT TERM
  echo "" >&2
  echo "Interrupted. Recorded run state is preserved at:" >&2
  echo "  $SHOWCASE_WORK_DIR" >&2
  manifest_state failed
  echo "Stop only this run's resources with:" >&2
  echo "  $0 --stop --work-dir $SHOWCASE_WORK_DIR" >&2
  exit 130
}

load_recorded_state() {
  local pointer state_dir state_file
  if [[ -n "$SHOWCASE_WORK_DIR" ]]; then
    state_dir="$SHOWCASE_WORK_DIR"
  else
    pointer="$(current_pointer)"
    if [[ ! -f "$pointer" ]]; then
      echo "ERROR: no recorded showcase run; set SHOWCASE_WORK_DIR or pass --work-dir" >&2
      return 1
    fi
    state_dir="$(<"$pointer")"
  fi
  state_file="$state_dir/state.env"
  if [[ ! -f "$state_file" ]]; then
    echo "ERROR: recorded showcase state is missing: $state_file" >&2
    return 1
  fi
  # The state file is mode 0600, created by this runner, and contains no secrets.
  # shellcheck disable=SC1090
  . "$state_file"
  NOVA_LOG_FILE="${NOVA_LOG_FILE:-$SHOWCASE_WORK_DIR/logs/nova-carter-ros2.log}"
}

# Preserve the complete ros2 launch output independently of Docker's log
# retention. This is best-effort so diagnostics never replace the original
# runner failure or prevent the remaining recorded resources from stopping.
capture_nova_ros2_log() {
  local reason="${1:-failure}"
  local log_file="${NOVA_LOG_FILE:-${SHOWCASE_WORK_DIR:-}/logs/nova-carter-ros2.log}"

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "+ save Nova Carter ROS 2 log to $log_file ($reason)"
    return 0
  fi
  if [[ -z "${NOVA_CONTAINER:-}" || -z "${SHOWCASE_WORK_DIR:-}" ]]; then
    return 0
  fi
  if ! docker inspect "$NOVA_CONTAINER" >/dev/null 2>&1; then
    return 0
  fi
  if ! mkdir -p "$(dirname "$log_file")"; then
    echo "WARNING: could not create Nova Carter log directory: $(dirname "$log_file")" >&2
    return 0
  fi
  if docker logs --timestamps "$NOVA_CONTAINER" > "$log_file" 2>&1; then
    echo "Saved Nova Carter ROS 2 log ($reason): $log_file"
  else
    echo "WARNING: could not save the complete Nova Carter ROS 2 log: $log_file" >&2
  fi
  return 0
}

capture_nova_ros2_log_on_failure() {
  local status=$?
  trap - EXIT
  if [[ $status -ne 0 ]]; then
    capture_nova_ros2_log failure
  fi
  exit "$status"
}

container_is_ignored() {
  local candidate="$1" ignored
  for ignored in "${IGNORED_CONTAINERS[@]}"; do
    if [[ "$candidate" == "$ignored" ]]; then
      return 0
    fi
  done
  return 1
}

check_no_running_containers() {
  local running name image line
  local -a blocking=()
  running="$(docker ps --format '{{.Names}}\t{{.Image}}')"
  while IFS=$'\t' read -r name image; do
    [[ -z "$name" ]] && continue
    line="${name}"$'\t'"${image}"
    if container_is_ignored "$name"; then
      echo "INFO: ignoring running Docker container: $line"
    else
      blocking+=("$line")
    fi
  done <<< "$running"
  if (( ${#blocking[@]} > 0 )); then
    echo "ERROR: non-ignored Docker containers are already running; use a dedicated clean host." >&2
    printf '%s\n' "${blocking[@]}" >&2
    return 1
  fi
}

run_doctor() {
  local ignored_container
  local args=(
    --skills-dir "$SKILLS_DIR"
    --asset-dir "$SHOWCASE_DIR/assets"
    --bring-up-cloud "$CLOUD_UPSTREAM_ENTRYPOINT"
    --change-map "$CHANGE_MAP"
    --change-fleet "$CHANGE_FLEET"
    --isaac-send "$ISAAC_SEND"
    --isaac-sim-dir "$ISAAC_SIM_DIR"
    --warehouse-usd-uri "$WAREHOUSE_USD_URI"
    --nova-carter-image "$NOVA_CARTER_IMAGE"
    --robot-name "$ROBOT_NAME"
    --isaac-python-port "$ISAAC_PYTHON_PORT"
  )
  for ignored_container in "${IGNORED_CONTAINERS[@]}"; do
    args+=(--ignore-container "$ignored_container")
  done
  python3 "$DOCTOR" host "${args[@]}"
}

prepare_runtime() {
  local fleet_json
  fleet_json="[{\"name\":\"${ROBOT_NAME}\",\"sim\":false}]"

  run mkdir -p "$MC_WORKSPACE/app/config/maps" "$RUNTIME_MAP_DIR"

  run cp "$SHOWCASE_DIR/assets/carter_warehouse_navigation.png" \
    "$RUNTIME_MAP_DIR/carter_warehouse_navigation.png"
  run cp "$SHOWCASE_DIR/assets/carter_warehouse_navigation.yaml" \
    "$RUNTIME_MAP_DIR/carter_warehouse_navigation.yaml"

  run env MC_WORKSPACE="$MC_WORKSPACE" python3 "$CHANGE_MAP" \
    --source-type file \
    --source "/tmp/config/maps/carter_warehouse_navigation.png" \
    --metadata-yaml "/tmp/config/maps/carter_warehouse_navigation.yaml"

  run cp "$SHOWCASE_DIR/assets/carter_warehouse_navigation.png" \
    "$MC_WORKSPACE/app/config/maps/carter_warehouse_navigation.png"
  run cp "$SHOWCASE_DIR/assets/carter_warehouse_navigation.yaml" \
    "$MC_WORKSPACE/app/config/maps/carter_warehouse_navigation.yaml"

  run env MC_WORKSPACE="$MC_WORKSPACE" python3 "$CHANGE_FLEET" \
    --fleet-json "$fleet_json"

  run chmod 0755 \
    "$RUNTIME_MAP_DIR" "$MC_WORKSPACE/app/config" "$MC_WORKSPACE/app/config/maps"
  run chmod 0644 \
    "$RUNTIME_MAP_DIR/carter_warehouse_navigation.png" \
    "$RUNTIME_MAP_DIR/carter_warehouse_navigation.yaml" \
    "$MC_WORKSPACE/app/config/maps/carter_warehouse_navigation.png" \
    "$MC_WORKSPACE/app/config/maps/carter_warehouse_navigation.yaml" \
    "$MC_WORKSPACE/app/config/defaults.yaml"

  run python3 "$SHOWCASE_HELPER" validate-maps \
    --source "$SHOWCASE_DIR/assets/carter_warehouse_navigation.png" \
    --source-yaml "$SHOWCASE_DIR/assets/carter_warehouse_navigation.yaml" \
    --mission-control "$MC_WORKSPACE/app/config/maps/carter_warehouse_navigation.png" \
    --mission-control-yaml "$MC_WORKSPACE/app/config/maps/carter_warehouse_navigation.yaml" \
    --carter "$RUNTIME_MAP_DIR/carter_warehouse_navigation.png" \
    --carter-yaml "$RUNTIME_MAP_DIR/carter_warehouse_navigation.yaml"
  run python3 "$SHOWCASE_HELPER" validate-defaults \
    --defaults "$MC_WORKSPACE/app/config/defaults.yaml" \
    --container-config-dir /tmp/config \
    --robot-name "$ROBOT_NAME"
}

ensure_image() {
  local image="$1"
  if docker image inspect "$image" >/dev/null 2>&1; then
    return 0
  fi
  if [[ $PULL_IMAGES -eq 1 ]]; then
    run docker pull "$image"
  else
    echo "ERROR: required image is not local and --no-pull was set: $image" >&2
    return 1
  fi
}

required_runtime_images() {
  printf '%s\n' \
    "$MOSQUITTO_IMAGE" \
    "$POSTGRES_IMAGE" \
    "$MISSION_DATABASE_IMAGE" \
    "$MISSION_DISPATCH_IMAGE" \
    "$WPG_IMAGE" \
    "$CUOPT_IMAGE" \
    "$MISSION_CONTROL_IMAGE" \
    "$NOVA_CARTER_IMAGE"
}

prefetch_runtime_images() {
  local image
  local -a images=()
  mapfile -t images < <(required_runtime_images | awk '!seen[$0]++')

  if [[ $DRY_RUN -eq 1 ]]; then
    for image in "${images[@]}"; do
      if [[ $PULL_IMAGES -eq 1 ]]; then
        print_command docker pull "$image"
      else
        echo "+ require local image: $image"
      fi
    done
    return 0
  fi

  if [[ $PULL_IMAGES -eq 0 ]]; then
    for image in "${images[@]}"; do
      ensure_image "$image"
    done
    return 0
  fi

  echo "Pulling ${#images[@]} showcase images with four concurrent workers."
  printf '%s\0' "${images[@]}" | xargs -0 -n1 -P4 docker pull
}

start_cloud_stack() {
  run env \
    MC_WORKSPACE="$MC_WORKSPACE" \
    COMPOSE_PROJECT_NAME="$COMPOSE_PROJECT_NAME" \
    "$BRING_UP_CLOUD"
}

# Return true when a Docker container exists and is running.
container_is_running() {
  [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" == "true" ]]
}

# Fail fast when a compose service exited after startup.
require_compose_container_running() {
  local service="$1"
  local name="${COMPOSE_PROJECT_NAME}-${service}-1"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "+ check compose service running: $service"
    return 0
  fi
  if ! container_is_running "$name"; then
    echo "ERROR: compose service is not running: $service ($name)" >&2
    docker logs --tail 80 "$name" >&2 || true
    return 1
  fi
}

# Wait for a TCP port to accept connections.
wait_tcp() {
  local host="$1"
  local port="$2"
  local timeout_s="$3"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "+ wait for TCP port: $host:$port"
    return 0
  fi
  local start_ts
  start_ts="$(date +%s)"
  while true; do
    if (echo >"/dev/tcp/$host/$port") >/dev/null 2>&1; then
      echo "Port ready: $host:$port"
      return 0
    fi
    if (( "$(date +%s)" - start_ts >= timeout_s )); then
      echo "ERROR: timeout waiting for $host:$port" >&2
      return 1
    fi
    sleep 2
  done
}

check_mqtt_websocket_listener() {
  local host="$1"
  local port="$2"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "+ check MQTT websocket listener: $host:$port"
    return 0
  fi
  python3 - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
key = "dGhlIHNhbXBsZSBub25jZQ=="  # gitleaks:allow - RFC 6455 example handshake nonce, not a credential
for path in ("/mqtt", "/"):
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "Sec-WebSocket-Protocol: mqtt\r\n\r\n"
    ).encode()
    try:
        with socket.create_connection((host, port), timeout=3) as sock:
            sock.sendall(request)
            response = sock.recv(4096).decode("latin1", "replace")
    except OSError:
        continue
    if " 101 " in response.split("\r\n", 1)[0]:
        print(f"MQTT websocket ready: {host}:{port}{path}")
        raise SystemExit(0)
print(f"ERROR: {host}:{port} did not accept an MQTT websocket handshake", file=sys.stderr)
raise SystemExit(1)
PY
}

wait_cloud_ready() {
  local service
  for service in mosquitto postgres mission-database mission-dispatch wpg cuopt mission-control; do
    require_compose_container_running "$service"
  done
  wait_tcp 127.0.0.1 "$MQTT_REQUIRED_PORT" 120
  wait_tcp 127.0.0.1 "$DATABASE_CONTROLLER_PORT" 180
  wait_tcp 127.0.0.1 "$MC_PORT" 180
  check_mqtt_websocket_listener 127.0.0.1 "$MQTT_REQUIRED_PORT"
}

# Launch the locally installed Isaac Sim with its GUI window. Everything
# downstream reaches the simulator only through the isaac-sim-remote Python
# server, so this is the single transport-specific step.
start_isaac_sim() {
  local launcher="$ISAAC_SIM_DIR/isaac-sim.sh"
  if [[ ! -x "$launcher" ]]; then
    echo "ERROR: Isaac Sim launcher is not executable: $launcher" >&2
    echo "       Set ISAAC_SIM_DIR to a local Isaac Sim installation root." >&2
    return 1
  fi
  if [[ $DRY_RUN -eq 1 ]]; then
    print_command "$launcher" --no-ros-env \
      --enable isaacsim.ros2.bridge \
      --enable isaacsim.code_editor.python_server \
      --/exts/isaacsim.code_editor.python_server/port="$ISAAC_PYTHON_PORT"
    return 0
  fi

  print_command "$launcher" --no-ros-env \
    --enable isaacsim.ros2.bridge \
    --enable isaacsim.code_editor.python_server \
    --/exts/isaacsim.code_editor.python_server/port="$ISAAC_PYTHON_PORT"

  # The remote payloads read their configuration from this environment, the
  # same contract the container build previously satisfied with -e flags.
  env \
    ACCEPT_EULA=Y \
    PRIVACY_CONSENT=Y \
    OMNI_ENV_PRIVACY_CONSENT=Y \
    ROS_DOMAIN_ID="$SHOWCASE_ROS_DOMAIN_ID" \
    ROS_LOCALHOST_ONLY="$SHOWCASE_ROS_LOCALHOST_ONLY" \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    LD_LIBRARY_PATH="$ISAAC_SIM_DIR/exts/isaacsim.ros2.core/humble/lib:${LD_LIBRARY_PATH:-}" \
    WAREHOUSE_USD_URI="$WAREHOUSE_USD_URI" \
    "$launcher" --no-ros-env \
    --enable isaacsim.ros2.bridge \
    --enable isaacsim.code_editor.python_server \
    --/exts/isaacsim.ros2.bridge/ros_distro=humble \
    --/exts/isaacsim.ros2.bridge/publish_without_verification=1 \
    --/exts/isaacsim.code_editor.python_server/host=127.0.0.1 \
    --/exts/isaacsim.code_editor.python_server/port="$ISAAC_PYTHON_PORT" \
    >"$ISAAC_LOG_FILE" 2>&1 &
  echo $! > "$ISAAC_PID_FILE"
  echo "isaac_sim_pid=$(<"$ISAAC_PID_FILE") log=$ISAAC_LOG_FILE"
}

isaac_sim_is_running() {
  local pid
  [[ -f "$ISAAC_PID_FILE" ]] || return 1
  pid="$(<"$ISAAC_PID_FILE")"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

# Wait for an actual Python-server response. A published Docker port can accept
# TCP connections before Isaac Sim has finished loading the extension and app.
wait_isaac_ready() {
  local timeout_s="${1:-300}"
  local start_ts
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "+ wait for Isaac Sim Python server response"
    return 0
  fi
  start_ts="$(date +%s)"
  while true; do
    if ! isaac_sim_is_running; then
      echo "ERROR: Isaac Sim exited during startup" >&2
      tail -n 120 "$ISAAC_LOG_FILE" >&2 || true
      return 1
    fi
    if python3 "$ISAAC_SEND" --port "$ISAAC_PYTHON_PORT" --timeout 10 \
      'print("ISAAC_REMOTE_READY")' >/dev/null 2>&1; then
      echo "Isaac Sim Python server ready."
      return 0
    fi
    if (( "$(date +%s)" - start_ts >= timeout_s )); then
      echo "ERROR: timeout waiting for Isaac Sim Python server" >&2
      return 1
    fi
    sleep 3
  done
}

# Restore the Isaac Sim stage, robot pose, and viewport camera.
restore_isaac() {
  wait_isaac_ready 300
  run python3 "$ISAAC_SEND" \
    --port "$ISAAC_PYTHON_PORT" \
    --timeout 900 \
    --file "$RESTORE_ISAAC"
}

start_nova_carter() {
  run docker run -d --name "$NOVA_CONTAINER" \
    --label com.nvidia.mission-control-showcase=true \
    --label "com.nvidia.mission-control-showcase.run=$COMPOSE_PROJECT_NAME" \
    --network host \
    --gpus "device=${SHOWCASE_GPU_DEVICE}" \
    --memory "$NOVA_CARTER_MEMORY_LIMIT" \
    --memory-swap "$NOVA_CARTER_MEMORY_SWAP_LIMIT" \
    --shm-size=1g \
    --cap-drop=ALL \
    --security-opt=no-new-privileges \
    -e "ROS_DOMAIN_ID=$SHOWCASE_ROS_DOMAIN_ID" \
    -e "ROS_LOCALHOST_ONLY=$SHOWCASE_ROS_LOCALHOST_ONLY" \
    -e RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    -e FASTDDS_BUILTIN_TRANSPORTS=UDPv4 \
    -v "$RUNTIME_MAP_DIR:/maps:ro" \
    "$NOVA_CARTER_IMAGE" /bin/bash -lc \
    "$ROS_SETUP_SNIPPET && exec ros2 launch isaac_ros_vda5050_nav2_client_bringup isaac_ros_vda5050_nav2_client.launch.py use_sim_time:=True map:=/maps/carter_warehouse_navigation.yaml init_pose_x:=-3.3 init_pose_y:=1.3 init_pose_yaw:=0.0001 use_static_tf:=True launch_rviz:=False serial_number:=${ROBOT_NAME} ros_to_mqtt_name:=${ROBOT_NAME}_RosToMqttBridge mqtt_to_ros_name:=${ROBOT_NAME}_MqttToRosBridge mqtt_host_name:=localhost mqtt_transport:=${MQTT_TRANSPORT} mqtt_port:=${MQTT_REQUIRED_PORT} mqtt_username:=${MQTT_USERNAME} mqtt_password:=${MQTT_PASSWORD}"
}

verify_active_map_consumers() {
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "+ verify Mission Control config, WPG graph, Carter map argument, and mounted map SHA"
    return 0
  fi

  local source_sha carter_sha source_yaml_sha carter_yaml_sha container_command
  source_sha="$(sha256sum "$SHOWCASE_DIR/assets/$MAP_PNG" | cut -d " " -f1)"
  carter_sha="$(docker exec "$NOVA_CONTAINER" sha256sum "/maps/$MAP_PNG" | cut -d " " -f1)"
  source_yaml_sha="$(sha256sum "$SHOWCASE_DIR/assets/$MAP_YAML" | cut -d " " -f1)"
  carter_yaml_sha="$(docker exec "$NOVA_CONTAINER" sha256sum "/maps/$MAP_YAML" | cut -d " " -f1)"
  if [[ "$source_sha" != "$EXPECTED_MAP_SHA256" || "$carter_sha" != "$source_sha" || "$carter_yaml_sha" != "$source_yaml_sha" ]]; then
    echo "ERROR: Carter mounted map does not match the bundled map" >&2
    echo "skill_sha256=$source_sha" >&2
    echo "carter_sha256=$carter_sha" >&2
    echo "skill_metadata_sha256=$source_yaml_sha" >&2
    echo "carter_metadata_sha256=$carter_yaml_sha" >&2
    return 1
  fi

  container_command="$(docker inspect --format "{{json .Config.Cmd}}" "$NOVA_CONTAINER")"
  if [[ "$container_command" != *"map:=/maps/$MAP_YAML"* ]]; then
    echo "ERROR: Carter was not configured with /maps/$MAP_YAML" >&2
    return 1
  fi

  run python3 "$SHOWCASE_HELPER" validate-wpg \
    --wpg-port "$WPG_PORT" \
    --map-id "$MAP_ID"
  echo "active_map_verified map_id=$MAP_ID sha256=$source_sha"
}

# Start the map-to-odom transform that the Nova Carter launch did not provide reliably.
start_static_tf() {
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "+ ensure map -> odom static transform in $NOVA_CONTAINER"
    return 0
  fi
  if timeout 15 docker exec "$NOVA_CONTAINER" bash -lc "
    $ROS_SETUP_SNIPPET
    timeout 5 ros2 run tf2_ros tf2_echo map base_link 2>/dev/null | grep -q 'Translation:'
  "; then
    echo "map -> base_link already resolves; no additional static transform needed."
    return 0
  fi
  docker exec -d "$NOVA_CONTAINER" bash -lc "
    $ROS_SETUP_SNIPPET
    exec ros2 run tf2_ros static_transform_publisher \
      --x 0 --y 0 --z 0 --yaw 0 --pitch 0 --roll 0 \
      --frame-id map --child-frame-id odom \
      --ros-args -r __node:=showcase_map_to_odom_static_tf
  "
}

# Nav2 subscribes to /scan. Isaac Sim 6.1 publishes the 2D LiDAR under the
# sensor's own namespace, so relay whatever LaserScan topic exists onto /scan
# rather than depending on the scene graph carrying a particular topic name.
scan_topic_probe() {
  local wait_s="$1"
  timeout "$2" docker exec -i "$NOVA_CONTAINER" \
    bash -lc "$ROS_SETUP_SNIPPET; exec python3 - $wait_s" <<'PYEOF'
import sys
import time

import rclpy

rclpy.init()
node = rclpy.create_node("showcase_scan_topic_probe")
deadline = time.monotonic() + float(sys.argv[1])
topics = []
while time.monotonic() < deadline:
    topics = [
        name
        for name, types in node.get_topic_names_and_types()
        if "sensor_msgs/msg/LaserScan" in types
    ]
    if topics:
        break
    rclpy.spin_once(node, timeout_sec=0.5)
if not topics:
    sys.exit(1)
print("/scan" if "/scan" in topics else sorted(topics)[0], end="")
PYEOF
}

ensure_scan_topic() {
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "+ ensure a sensor_msgs/msg/LaserScan is published on /scan"
    return 0
  fi

  local source_topic
  if ! source_topic="$(scan_topic_probe 90 120)"; then
    echo "ERROR: no sensor_msgs/msg/LaserScan topic is being published." >&2
    echo "       The Isaac 2D LiDAR render product is not enabled; check the" >&2
    echo "       set_graph_attr lines in the restore_isaac output." >&2
    return 1
  fi

  if [[ "$source_topic" == "/scan" ]]; then
    echo "scan_topic=/scan (no relay needed)"
    return 0
  fi

  echo "scan_topic=$source_topic relaying to /scan"
  docker exec -i "$NOVA_CONTAINER" \
    bash -lc "cat > /tmp/showcase_scan_relay.py" <<'RELAYEOF'
import sys

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

source, target = sys.argv[1], sys.argv[2]
rclpy.init()
node = rclpy.create_node("showcase_scan_relay")
publisher = node.create_publisher(LaserScan, target, qos_profile_sensor_data)
node.create_subscription(
    LaserScan, source, publisher.publish, qos_profile_sensor_data
)
rclpy.spin(node)
RELAYEOF
  docker exec -d "$NOVA_CONTAINER" bash -lc \
    "$ROS_SETUP_SNIPPET; exec python3 /tmp/showcase_scan_relay.py '$source_topic' /scan"

  local relayed
  if ! relayed="$(scan_topic_probe 30 60)" || [[ "$relayed" != "/scan" ]]; then
    echo "ERROR: relay from $source_topic to /scan did not come up" >&2
    return 1
  fi
  echo "scan_relay_ready=/scan"
}

# Require a usable navigation stack, not only Mission Control registration.
wait_nav2_ready() {
  local timeout_s="${1:-240}"
  local attempt_status oom_killed start_ts
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "+ wait for /clock, odom, TF, and active Nav2 lifecycle nodes"
    return 0
  fi
  start_ts="$(date +%s)"
  while true; do
    if ! isaac_sim_is_running; then
      echo "ERROR: Isaac Sim exited while waiting for Nav2 readiness" >&2
      tail -n 120 "$ISAAC_LOG_FILE" >&2 || true
      return 1
    fi
    if ! container_is_running "$NOVA_CONTAINER"; then
      echo "ERROR: Carter SIL exited while waiting for Nav2 readiness" >&2
      docker logs --tail 120 "$NOVA_CONTAINER" >&2 || true
      return 1
    fi
    oom_killed="$(docker inspect -f '{{.State.OOMKilled}}' "$NOVA_CONTAINER")"
    if [[ "$oom_killed" == "true" ]]; then
      echo "ERROR: Carter SIL recorded a cgroup OOM kill during Nav2 startup" >&2
      docker logs --tail 120 "$NOVA_CONTAINER" >&2 || true
      return 1
    fi
    if timeout 30 docker exec "$NOVA_CONTAINER" bash -lc "
      $ROS_SETUP_SNIPPET
      (timeout 5 ros2 topic echo /clock --once >/dev/null 2>&1 || exit 1) &&
      (timeout 5 ros2 topic echo /chassis/odom --once >/dev/null 2>&1 || exit 1) &&
      timeout 5 ros2 run tf2_ros tf2_echo map base_link 2>/dev/null | grep -q 'Translation:' &&
      timeout 5 ros2 lifecycle get /planner_server 2>/dev/null | grep -q 'active \[3\]' &&
      timeout 5 ros2 lifecycle get /controller_server 2>/dev/null | grep -q 'active \[3\]' &&
      timeout 5 ros2 lifecycle get /bt_navigator 2>/dev/null | grep -q 'active \[3\]'
    "; then
      echo "Nav2 basics ready: clock, odom, TF, planner, controller, and navigator."
      return 0
    else
      attempt_status=$?
      if [[ $attempt_status -eq 124 ]]; then
        echo "ERROR: a Nav2 ROS CLI readiness attempt exceeded 30 seconds" >&2
        return 1
      fi
    fi
    if (( "$(date +%s)" - start_ts >= timeout_s )); then
      echo "ERROR: Nav2 basics did not become ready within ${timeout_s}s" >&2
      docker exec "$NOVA_CONTAINER" bash -lc "
        $ROS_SETUP_SNIPPET
        for topic in /clock /chassis/odom /scan; do
          if timeout 5 ros2 topic echo \"\$topic\" --once >/dev/null 2>&1; then
            echo \"\$topic=message\"
          else
            echo \"\$topic=missing\"
          fi
        done
        if timeout 5 ros2 run tf2_ros tf2_echo map base_link 2>/dev/null | grep -q 'Translation:'; then
          echo 'map_to_base_link=ready'
        else
          echo 'map_to_base_link=missing'
        fi
        for node in planner_server controller_server bt_navigator; do
          state=\$(timeout 5 ros2 lifecycle get \"/\$node\" 2>&1 || true)
          echo \"\$node=\$state\"
        done
      " >&2 || true
      return 1
    fi
    sleep 3
  done
}

probe_runtime_consistency() {
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "+ verify advancing clock, LiDAR timestamp transformability, and costmap inclusion"
    return 0
  fi
  if timeout 75 docker exec -i "$NOVA_CONTAINER" \
    bash -lc "$ROS_SETUP_SNIPPET; exec python3 -" <<'PY'
import math
import time

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.time import Time
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

rclpy.init()
node = rclpy.create_node("showcase_runtime_consistency_probe")
buffer = Buffer()
listener = TransformListener(buffer, node)
messages = {
    "clock_first": None,
    "clock": None,
    "scan": None,
    "scan_topic": None,
    "costmap": None,
    "scan_candidates": [],
}

def clock_cb(msg):
    if messages["clock_first"] is None:
        messages["clock_first"] = msg.clock
    messages["clock"] = msg.clock

def scan_cb(msg, topic):
    messages["scan"] = msg
    messages["scan_topic"] = topic
    if msg.header.frame_id == "front_2d_lidar":
        messages["scan_candidates"].append((msg, topic))
        if len(messages["scan_candidates"]) > 100:
            del messages["scan_candidates"][:-100]

def costmap_cb(msg):
    messages["costmap"] = msg

node.create_subscription(Clock, "/clock", clock_cb, 10)
deadline = time.monotonic() + 20.0
scan_topics = []
costmap_topic = None
while time.monotonic() < deadline and (not scan_topics or costmap_topic is None):
    for name, types in node.get_topic_names_and_types():
        if "sensor_msgs/msg/LaserScan" in types and name not in scan_topics:
            scan_topics.append(name)
        if (
            costmap_topic is None
            and "nav_msgs/msg/OccupancyGrid" in types
            and "global_costmap" in name
            and name.endswith("/costmap")
        ):
            costmap_topic = name
    rclpy.spin_once(node, timeout_sec=0.2)

if not scan_topics:
    raise RuntimeError("no sensor_msgs/msg/LaserScan topic is available")
if costmap_topic is None:
    raise RuntimeError("no global Nav2 costmap OccupancyGrid topic is available")

scan_subscriptions = [
    node.create_subscription(
        LaserScan,
        topic,
        lambda msg, topic=topic: scan_cb(msg, topic),
        qos_profile_sensor_data,
    )
    for topic in scan_topics
]
node.create_subscription(OccupancyGrid, costmap_topic, costmap_cb, 10)
deadline = time.monotonic() + 50.0
transform_ready = False
while time.monotonic() < deadline:
    rclpy.spin_once(node, timeout_sec=0.2)
    first = messages["clock_first"]
    latest = messages["clock"]
    advanced = (
        first is not None
        and latest is not None
        and (latest.sec, latest.nanosec) > (first.sec, first.nanosec)
    )
    if not advanced or messages["costmap"] is None:
        continue
    clock_s = latest.sec + latest.nanosec / 1e9
    for candidate, topic in reversed(messages["scan_candidates"]):
        candidate_s = candidate.header.stamp.sec + candidate.header.stamp.nanosec / 1e9
        if abs(clock_s - candidate_s) > 1.0:
            continue
        candidate_time = Time.from_msg(candidate.header.stamp)
        if buffer.can_transform("map", candidate.header.frame_id, candidate_time):
            messages["scan"] = candidate
            messages["scan_topic"] = topic
            transform_ready = True
            break
    if transform_ready:
        break

first = messages["clock_first"]
clock = messages["clock"]
scan = messages["scan"]
scan_topic = messages["scan_topic"]
costmap = messages["costmap"]
if first is None or clock is None or (clock.sec, clock.nanosec) <= (first.sec, first.nanosec):
    raise RuntimeError("simulation /clock did not advance")
if scan is None:
    raise RuntimeError(f"no live LiDAR sample arrived on {scan_topic}")
if scan.header.frame_id != "front_2d_lidar":
    raise RuntimeError(
        f"{scan_topic} frame is {scan.header.frame_id!r}; "
        "expected 'front_2d_lidar'"
    )
if costmap is None:
    raise RuntimeError(f"no live costmap arrived on {costmap_topic}")

clock_s = clock.sec + clock.nanosec / 1e9
scan_s = scan.header.stamp.sec + scan.header.stamp.nanosec / 1e9
delta_s = abs(clock_s - scan_s)
if delta_s > 1.0:
    raise RuntimeError(
        f"LiDAR stamp differs from simulation clock by {delta_s:.3f}s "
        f"(clock={clock_s:.3f}, scan={scan_s:.3f})"
    )

scan_time = Time.from_msg(scan.header.stamp)
if not transform_ready:
    raise RuntimeError(
        f"TF cannot transform {scan.header.frame_id} -> map at latest LiDAR "
        f"stamp {scan_s:.3f} before the readiness deadline"
    )

transform = buffer.lookup_transform("map", "base_link", Time())
x = transform.transform.translation.x
y = transform.transform.translation.y
info = costmap.info
column = math.floor((x - info.origin.position.x) / info.resolution)
row = math.floor((y - info.origin.position.y) / info.resolution)
if not (0 <= column < info.width and 0 <= row < info.height):
    raise RuntimeError(
        f"robot pose ({x:.3f}, {y:.3f}) is outside costmap "
        f"origin=({info.origin.position.x:.3f}, {info.origin.position.y:.3f}) "
        f"size=({info.width}, {info.height}) resolution={info.resolution}"
    )

print(
    "runtime_consistency_ok "
    f"clock_delta_s={delta_s:.3f} scan_topic={scan_topic} "
    f"scan_frame={scan.header.frame_id} costmap_topic={costmap_topic} "
    f"robot_cell=({column},{row})"
)
node.destroy_node()
rclpy.shutdown()
PY
  then
    return 0
  fi
  if ! isaac_sim_is_running; then
    echo "ERROR: Isaac Sim exited during the runtime consistency probe" >&2
    tail -n 120 "$ISAAC_LOG_FILE" >&2 || true
    return 1
  fi
  if [[ "$(docker inspect -f '{{.State.OOMKilled}}' "$NOVA_CONTAINER" 2>/dev/null)" == "true" ]]; then
    echo "ERROR: Carter SIL recorded a cgroup OOM kill during the runtime consistency probe" >&2
    docker logs --tail 120 "$NOVA_CONTAINER" >&2 || true
    return 1
  fi
  echo "ERROR: runtime consistency probe failed or exceeded 75 seconds" >&2
  return 1
}

# Wait until Mission Control sees Carter online, idle, and error-free.
wait_mc_robot() {
  run python3 "$SHOWCASE_HELPER" wait-robot \
    --robot-name "$ROBOT_NAME" \
    --database-controller-port "$DATABASE_CONTROLLER_PORT" \
    --timeout 180
}

# Require the SIL parent and the navigation processes needed to execute a
# mission. A ros2 launch parent can remain alive after cgroup OOM kills its
# children, so container-running status alone is not a health signal.
require_nova_carter_healthy() {
  local quiet="${1:-0}"
  local oom_killed
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "+ verify Carter SIL has no OOM event and required Nav2 processes are alive"
    return 0
  fi
  if ! container_is_running "$NOVA_CONTAINER"; then
    echo "ERROR: Carter SIL container is not running: $NOVA_CONTAINER" >&2
    docker logs --tail 120 "$NOVA_CONTAINER" >&2 || true
    return 1
  fi
  oom_killed="$(docker inspect -f '{{.State.OOMKilled}}' "$NOVA_CONTAINER")"
  if [[ "$oom_killed" == "true" ]]; then
    echo "ERROR: Carter SIL recorded a cgroup OOM kill: $NOVA_CONTAINER" >&2
    docker logs --tail 120 "$NOVA_CONTAINER" >&2 || true
    return 1
  fi
  if ! docker exec "$NOVA_CONTAINER" bash -lc '
    required=(
      "^/opt/ros/humble/lib/nav2_map_server/map_server "
      "^/opt/ros/humble/lib/nav2_amcl/amcl "
      "^/opt/ros/humble/lib/nav2_controller/controller_server "
      "^/opt/ros/humble/lib/nav2_planner/planner_server "
      "^/opt/ros/humble/lib/nav2_bt_navigator/bt_navigator "
      "^/opt/ros/humble/lib/isaac_ros_vda5050_nav2_client/vda5050_nav2_client "
    )
    for pattern in "${required[@]}"; do
      if ! pgrep -f -- "$pattern" >/dev/null; then
        echo "missing_required_process=$pattern" >&2
        exit 1
      fi
    done
  '; then
    echo "ERROR: Carter SIL is missing a required navigation process" >&2
    docker logs --tail 120 "$NOVA_CONTAINER" >&2 || true
    return 1
  fi
  if [[ "$quiet" != "1" ]]; then
    echo "Carter SIL healthy: no OOM event and required Nav2 processes are alive."
  fi
}

# Submit the known-good Mission Control route mission for Carter.
submit_known_good_mission() {
  require_nova_carter_healthy
  DEMO_MISSION_ID="showcase-${COMPOSE_PROJECT_NAME}-$(date -u +%Y%m%dT%H%M%SZ)"
  run python3 "$SHOWCASE_HELPER" submit-mission \
    --robot-name "$ROBOT_NAME" \
    --mission-id "$DEMO_MISSION_ID" \
    --mc-api-port "$MC_PORT"
}

# Wait until the demo mission runs and Carter returns to idle.
wait_demo_complete() {
  local waiter_pid
  if [[ $DRY_RUN -eq 1 ]]; then
    run python3 "$SHOWCASE_HELPER" wait-demo-complete \
      --robot-name "$ROBOT_NAME" \
      --mission-id "$DEMO_MISSION_ID" \
      --database-controller-port "$DATABASE_CONTROLLER_PORT" \
      --timeout 900
    return 0
  fi
  print_command python3 "$SHOWCASE_HELPER" wait-demo-complete \
    --robot-name "$ROBOT_NAME" \
    --mission-id "$DEMO_MISSION_ID" \
    --database-controller-port "$DATABASE_CONTROLLER_PORT" \
    --timeout 900
  python3 "$SHOWCASE_HELPER" wait-demo-complete \
    --robot-name "$ROBOT_NAME" \
    --mission-id "$DEMO_MISSION_ID" \
    --database-controller-port "$DATABASE_CONTROLLER_PORT" \
    --timeout 900 &
  waiter_pid=$!
  while kill -0 "$waiter_pid" 2>/dev/null; do
    if ! require_nova_carter_healthy 1; then
      kill "$waiter_pid" 2>/dev/null || true
      wait "$waiter_pid" 2>/dev/null || true
      echo "ERROR: demo aborted because Carter SIL became unhealthy" >&2
      return 1
    fi
    sleep 3
  done
  if ! wait "$waiter_pid"; then
    return 1
  fi
  require_nova_carter_healthy
}

stop_recorded_showcase() {
  local name pid
  load_recorded_state
  trap capture_nova_ros2_log_on_failure EXIT
  for name in "$NOVA_CONTAINER"; do
    if docker inspect "$name" >/dev/null 2>&1 && container_is_running "$name"; then
      run docker stop "$name"
    fi
  done
  capture_nova_ros2_log --stop

  # Isaac Sim is a local process now; SIGTERM lets Kit shut down cleanly.
  if [[ -n "${ISAAC_PID_FILE:-}" && -f "$ISAAC_PID_FILE" ]]; then
    pid="$(<"$ISAAC_PID_FILE")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      run kill -TERM "$pid"
      echo "Sent SIGTERM to Isaac Sim pid $pid."
    fi
  fi

  export CONFIG_DIR="$MC_WORKSPACE/app/config"
  if (( CLOUD_RESOURCES_RESOLVED )); then
    load_cloud_defaults
    export INIT_DB_SH="$CLOUD_INIT_DB"
  fi
  if (( CLOUD_RESOURCES_RESOLVED )) &&
     [[ -f "$MC_WORKSPACE/docker-compose/bringup_services.yaml" ]]; then
    run docker compose \
      -p "$COMPOSE_PROJECT_NAME" \
      -f "$MC_WORKSPACE/docker-compose/bringup_services.yaml" \
      --profile enable_mission_control \
      stop
  fi
  echo "Stopped showcase resources recorded in $SHOWCASE_WORK_DIR; files and containers remain recoverable."
}

# These three ship with this package, so they are checked unconditionally.
require_file "$RESTORE_ISAAC"
require_file "$SHOWCASE_HELPER"
require_file "$DOCTOR"

# --stop resolves only the cloud file paths, and tolerates failure: a run whose
# upstream checkout moved must still be able to stop what it recorded. Every
# other path needs the adapters and fails loudly without them.
if [[ $STOP -eq 1 ]]; then
  if ! resolve_cloud_resources; then
    echo "WARNING: could not resolve bring-up-cloud-stack resources." >&2
    echo "         Stopping the resources this run recorded anyway; the compose" >&2
    echo "         teardown will be skipped." >&2
  fi
else
  resolve_runtime_entrypoints || exit 1
  load_cloud_defaults

  if [[ "$MQTT_TRANSPORT" != "websockets" ]]; then
    echo "ERROR: the packaged showcase requires websocket MQTT" >&2
    exit 1
  fi
fi

ROS_SETUP_SNIPPET='if [[ -n "${ROS_DISTRO:-}" ]]; then source "/opt/ros/${ROS_DISTRO}/setup.bash"; else setup="$(find /opt/ros -maxdepth 2 -name setup.bash | head -n 1)"; [[ -n "$setup" ]] || { echo "ERROR: no ROS setup.bash under /opt/ros" >&2; exit 1; }; source "$setup"; fi'

if ! [[ "$SHOWCASE_ROS_DOMAIN_ID" =~ ^[0-9]+$ ]] ||
   (( SHOWCASE_ROS_DOMAIN_ID > 232 )); then
  echo "ERROR: SHOWCASE_ROS_DOMAIN_ID must be an integer from 0 through 232" >&2
  exit 2
fi
if [[ "$SHOWCASE_ROS_LOCALHOST_ONLY" != "0" &&
      "$SHOWCASE_ROS_LOCALHOST_ONLY" != "1" ]]; then
  echo "ERROR: SHOWCASE_ROS_LOCALHOST_ONLY must be 0 or 1" >&2
  exit 2
fi

if [[ $STOP -eq 1 ]]; then
  stop_recorded_showcase
  exit 0
fi

if [[ $PREFLIGHT -eq 1 ]]; then
  run_doctor
  exit 0
fi

if [[ $DRY_RUN -eq 1 ]]; then
  prefetch_runtime_images
  initialize_work_dir
  prepare_runtime
  manifest_state starting-cloud
  start_cloud_stack
  manifest_state starting-isaac
  start_isaac_sim
  restore_isaac
  wait_cloud_ready
  manifest_state starting-carter
  start_nova_carter
  verify_active_map_consumers
  start_static_tf
  ensure_scan_topic
  wait_nav2_ready
  probe_runtime_consistency
  require_nova_carter_healthy
  manifest_state ready
  wait_mc_robot
  if [[ $DEMO -eq 1 ]]; then
    submit_known_good_mission
    wait_demo_complete
  fi
  exit 0
fi

run_doctor
check_no_running_containers
prefetch_runtime_images
initialize_work_dir
trap capture_nova_ros2_log_on_failure EXIT
prepare_runtime
start_cloud_stack
start_isaac_sim
restore_isaac
wait_cloud_ready
start_nova_carter
verify_active_map_consumers
start_static_tf
ensure_scan_topic
wait_nav2_ready 240
probe_runtime_consistency
require_nova_carter_healthy
wait_mc_robot

if [[ $DEMO -eq 1 ]]; then
  manifest_state mission-running
  submit_known_good_mission
  wait_demo_complete
fi

echo "Showcase stack is ready."
echo "robot_name=$ROBOT_NAME"
echo "warehouse_usd_uri=$WAREHOUSE_USD_URI"
echo "isaac_sim_pid_file=$ISAAC_PID_FILE"
