#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Collect read-only evidence used to run and report NVSHMEM performance tests.
# Exit 0 means the core evidence categories were found; exit 2 means the
# output is still useful but one or more categories are incomplete.

set -u

usage() {
    printf '%s\n' \
        'Usage: collect-performance-environment.sh [--prefix PATH]' \
        '' \
        'Collect labeled, read-only NVSHMEM performance-environment evidence.' \
        'Run this on each participating compute node. No benchmark is run.'
}

diagnostics=()
last_output=""
last_status=0

section() {
    printf '\n[%s]\n' "$1"
}

note_missing() {
    diagnostics+=("$1")
}

run_probe() {
    local label=$1
    shift
    printf '%s:\n' "$label"
    "$@" 2>&1
    local rc=$?
    printf '%s_exit_status: %d\n' "$label" "$rc"
    return "$rc"
}

# Performance-specific extension: retain output needed for fallback handling and summaries.
capture_probe() {
    local label=$1
    shift
    printf '%s:\n' "$label"
    last_output="$("$@" 2>&1)"
    last_status=$?
    if [[ -n "$last_output" ]]; then
        printf '%s\n' "$last_output"
    else
        printf 'output: (empty)\n'
    fi
    printf '%s_exit_status: %d\n' "$label" "$last_status"
    return "$last_status"
}

print_path() {
    local command_name=$1
    local resolved
    resolved="$(command -v "$command_name" 2>/dev/null || true)"
    printf '%s: %s\n' "$command_name" "${resolved:-unavailable}"
}

# Performance-specific extension: resolve both nvshmem-info and the NVSHMEM launcher.
resolve_nvshmem_tool() {
    local command_name=$1
    local path_variable=$2
    local source_variable=$3
    local candidate=''
    local source='unavailable'
    local absolute_candidate=''

    if [[ -n "$nvshmem_prefix" && -f "$nvshmem_prefix/bin/$command_name" &&
        -x "$nvshmem_prefix/bin/$command_name" ]]; then
        candidate="$nvshmem_prefix/bin/$command_name"
        source='resolved prefix'
    else
        candidate="$(type -P "$command_name" 2>/dev/null || true)"
        if [[ -n "$candidate" && -f "$candidate" && -x "$candidate" ]]; then
            source='PATH'
        else
            candidate=''
        fi
    fi

    if [[ -n "$candidate" ]]; then
        absolute_candidate="$(readlink -f -- "$candidate" 2>/dev/null || true)"
        [[ -n "$absolute_candidate" ]] && candidate=$absolute_candidate
    fi

    printf -v "$path_variable" '%s' "$candidate"
    printf -v "$source_variable" '%s' "$source"
}

os_release_value() {
    local key=$1
    if [[ -r /etc/os-release ]]; then
        sed -n "s/^${key}=//p" /etc/os-release | sed -n '1p' | tr -d '"'
    fi
}

prefix_arg=''
prefix_arg_set=0
while (($#)); do
    case "$1" in
        --prefix)
            if (($# < 2)) || [[ -z "$2" ]]; then
                printf 'error: --prefix requires PATH\n' >&2
                usage >&2
                exit 64
            fi
            prefix_arg=$2
            prefix_arg_set=1
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf 'error: unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 64
            ;;
    esac
done

nvshmem_prefix=''
prefix_source='unresolved'
if ((prefix_arg_set)); then
    nvshmem_prefix=$prefix_arg
    prefix_source='--prefix'
elif [[ -n "${NVSHMEM_PREFIX:-}" ]]; then
    nvshmem_prefix=$NVSHMEM_PREFIX
    prefix_source='NVSHMEM_PREFIX'
elif [[ -n "${NVSHMEM_HOME:-}" ]]; then
    nvshmem_prefix=$NVSHMEM_HOME
    prefix_source='NVSHMEM_HOME'
else
    info_on_path="$(type -P nvshmem-info 2>/dev/null || true)"
    if [[ -n "$info_on_path" ]]; then
        absolute_info_on_path="$(readlink -f -- "$info_on_path" 2>/dev/null || true)"
        [[ -n "$absolute_info_on_path" ]] && info_on_path=$absolute_info_on_path
        nvshmem_prefix="$(dirname "$(dirname "$info_on_path")")"
        prefix_source='nvshmem-info on PATH'
    fi
fi

nvshmem_info=''
nvshmem_info_source='unavailable'
nvshmrun_bin=''
nvshmrun_source='unavailable'
resolve_nvshmem_tool nvshmem-info nvshmem_info nvshmem_info_source
resolve_nvshmem_tool nvshmrun nvshmrun_bin nvshmrun_source

section collector
if ! hostname_value=$(hostname 2>/dev/null); then
    hostname_value='unknown'
fi
if ! effective_user=$(id -un 2>/dev/null); then
    effective_user="uid:$(id -u 2>/dev/null || printf unknown)"
fi
printf 'schema: nvshmem-performance-environment-v2\n'
printf 'timestamp_utc: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf 'hostname: %s\n' "$hostname_value"
printf 'effective_user: %s\n' "$effective_user"
printf 'kernel: %s\n' "$(uname -srmo 2>/dev/null || true)"
printf 'os_id: %s\n' "$(os_release_value ID)"
printf 'os_version: %s\n' "$(os_release_value VERSION_ID)"
printf 'cpu_arch: %s\n' "$(uname -m 2>/dev/null || true)"
if command -v lscpu >/dev/null 2>&1; then
    printf 'cpu_model: %s\n' "$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | sed -n '1p')"
    printf 'cpu_numa_nodes: %s\n' "$(lscpu 2>/dev/null | sed -n 's/^NUMA node(s):[[:space:]]*//p' | sed -n '1p')"
else
    printf 'cpu_model: unknown\n'
    printf 'cpu_numa_nodes: unknown\n'
fi

container_type=none
if [[ -f /.dockerenv ]]; then
    container_type=docker-compatible
elif [[ -n "${APPTAINER_NAME:-}" || -n "${SINGULARITY_NAME:-}" ]]; then
    container_type=apptainer-or-singularity
elif [[ -r /proc/1/cgroup ]] && grep -Eq 'docker|containerd|kubepods|libpod' /proc/1/cgroup 2>/dev/null; then
    container_type=container-cgroup
fi
printf 'container: %s\n' "$container_type"
printf 'prefix_source: %s\n' "$prefix_source"
if [[ -n "$nvshmem_prefix" ]]; then
    printf 'nvshmem_prefix: %s\n' "$nvshmem_prefix"
    if [[ ! -d "$nvshmem_prefix" || ! -r "$nvshmem_prefix" ]]; then
        note_missing "resolved NVSHMEM prefix is not a readable directory: $nvshmem_prefix"
    fi
else
    printf 'nvshmem_prefix: unresolved\n'
    note_missing 'exact NVSHMEM installation prefix is unresolved'
fi

section nvshmem-version
printf 'nvshmem_info: %s\n' "${nvshmem_info:-not found}"
printf 'nvshmem_info_source: %s\n' "$nvshmem_info_source"
printf 'NVSHMEM_PREFIX: %s\n' "${NVSHMEM_PREFIX:-unset}"
printf 'NVSHMEM_HOME: %s\n' "${NVSHMEM_HOME:-unset}"
if [[ -n "$nvshmem_info" ]]; then
    info_prefix_library_path=''
    if [[ "$nvshmem_info_source" == 'resolved prefix' ]]; then
        for libdir in "$nvshmem_prefix/lib" "$nvshmem_prefix/lib64"; do
            [[ -d "$libdir" ]] || continue
            if [[ -n "$info_prefix_library_path" ]]; then
                info_prefix_library_path+=":$libdir"
            else
                info_prefix_library_path=$libdir
            fi
        done
    fi
    printf 'nvshmem_info_prefix_library_path: %s\n' "${info_prefix_library_path:-none}"

    info_environment=(env LC_ALL=C)
    if [[ -n "$info_prefix_library_path" ]]; then
        info_ld_library_path=$info_prefix_library_path
        if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
            info_ld_library_path+=":$LD_LIBRARY_PATH"
        fi
        info_environment+=("LD_LIBRARY_PATH=$info_ld_library_path")
    fi

    capture_probe nvshmem_info_build_query "${info_environment[@]}" "$nvshmem_info" -n -b || true
    if ((last_status != 0)); then
        capture_probe nvshmem_info_version_query "${info_environment[@]}" "$nvshmem_info" -n || true
        if ((last_status != 0)); then
            note_missing 'nvshmem-info -n -b and the version-only fallback both failed.'
        fi
    fi
else
    note_missing 'nvshmem-info is unavailable; load the target NVSHMEM environment.'
fi

section cuda-and-gpus
print_path nvidia-smi
printf 'CUDA_HOME: %s\n' "${CUDA_HOME:-unset}"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
if command -v nvidia-smi >/dev/null 2>&1; then
    capture_probe gpu_inventory nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version --format=csv,noheader || true
    if ((last_status != 0)); then
        note_missing 'GPU inventory failed; run on an allocated GPU compute node.'
    else
        gpu_count="$(printf '%s\n' "$last_output" | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
        printf 'gpu_count: %s\n' "$gpu_count"
    fi
    if ! run_probe gpu_topology nvidia-smi topo -m; then
        note_missing 'nvidia-smi topo -m failed; GPU/NIC topology is incomplete.'
    fi
else
    note_missing 'nvidia-smi is unavailable; run on an allocated GPU compute node.'
fi

print_path nvcc
if command -v nvcc >/dev/null 2>&1; then
    run_probe nvcc_version nvcc --version || true
fi
if [[ -r /proc/driver/nvidia/version ]]; then
    printf 'driver_module: %s\n' "$(sed -n '1p' /proc/driver/nvidia/version)"
else
    printf 'driver_module: unavailable\n'
fi
if [[ -d /sys/module/nvidia_peermem ]]; then
    printf 'nvidia_peermem: loaded\n'
else
    printf 'nvidia_peermem: not-detected\n'
fi

section network-interfaces
if command -v ip >/dev/null 2>&1; then
    if ! run_probe network_links ip -brief link show; then
        note_missing 'ip -brief link show failed; general network-interface evidence is incomplete.'
    fi
else
    printf 'ip: unavailable\n'
    note_missing 'ip is unavailable; general network-interface evidence is incomplete.'
fi

section rdma-ports
shopt -s nullglob
port_paths=(/sys/class/infiniband/*/ports/*)
ports_found=0
for port_path in "${port_paths[@]}"; do
    [[ -d "$port_path" ]] || continue
    ports_found=$((ports_found + 1))

    hca_dir=${port_path%/ports/*}
    hca_name=${hca_dir##*/}
    port_number=${port_path##*/}
    state=unknown
    physical_state=unknown
    link_layer=unknown
    rate=unknown
    [[ -r "$port_path/state" ]] && state="$(<"$port_path/state")"
    [[ -r "$port_path/phys_state" ]] && physical_state="$(<"$port_path/phys_state")"
    [[ -r "$port_path/link_layer" ]] && link_layer="$(<"$port_path/link_layer")"
    [[ -r "$port_path/rate" ]] && rate="$(<"$port_path/rate")"

    pci_device=unknown
    device_path="$(readlink -f "$hca_dir/device" 2>/dev/null || true)"
    [[ -n "$device_path" ]] && pci_device=${device_path##*/}

    net_names=()
    net_paths=("$hca_dir"/device/net/*)
    for net_path in "${net_paths[@]}"; do
        net_names+=("${net_path##*/}")
    done
    netdev=none
    if ((${#net_names[@]} > 0)); then
        netdev="$(IFS=,; printf '%s' "${net_names[*]}")"
    fi

    printf 'hca: %q port: %q pci: %q state: %q physical_state: %q link_layer: %q rate: %q netdev: %q\n' \
        "$hca_name" "$port_number" "$pci_device" "$state" "$physical_state" \
        "$link_layer" "$rate" "$netdev"
done
printf 'ports_found: %d\n' "$ports_found"
if ((ports_found == 0)); then
    note_missing 'No RDMA ports were visible under /sys/class/infiniband; this may be valid for a non-RDMA system.'
fi

section rdma-tools
print_path rdma
print_path ibv_devinfo
print_path ibv_devices
if command -v rdma >/dev/null 2>&1; then
    run_probe rdma_links rdma link show || true
fi
if command -v ibv_devices >/dev/null 2>&1; then
    run_probe verbs_devices ibv_devices || true
fi
if command -v ibv_devinfo >/dev/null 2>&1; then
    run_probe verbs_device_details ibv_devinfo -v || true
fi

section launcher-and-allocation
printf 'nvshmrun: %s\n' "${nvshmrun_bin:-unavailable}"
printf 'nvshmrun_source: %s\n' "$nvshmrun_source"
for launcher in srun mpirun mpiexec oshrun; do
    print_path "$launcher"
done
if command -v srun >/dev/null 2>&1; then
    run_probe srun_version srun --version || true
fi
if command -v mpirun >/dev/null 2>&1; then
    run_probe mpirun_version mpirun --version || true
fi
printf 'SLURM_JOB_ID: %s\n' "${SLURM_JOB_ID:-unset}"
printf 'SLURM_JOB_NODELIST: %s\n' "${SLURM_JOB_NODELIST:-unset}"
printf 'SLURM_NNODES: %s\n' "${SLURM_NNODES:-unset}"
printf 'SLURM_NTASKS: %s\n' "${SLURM_NTASKS:-unset}"
printf 'SLURM_TASKS_PER_NODE: %s\n' "${SLURM_TASKS_PER_NODE:-unset}"
printf 'SLURM_GPUS: %s\n' "${SLURM_GPUS:-unset}"
printf 'SLURM_GPUS_ON_NODE: %s\n' "${SLURM_GPUS_ON_NODE:-unset}"

section relevant-environment
env | LC_ALL=C sort | awk -F= '
    $1 ~ /^NVSHMEM_/ ||
    $1 == "CUDA_HOME" ||
    $1 == "CUDA_VISIBLE_DEVICES" ||
    $1 == "LD_LIBRARY_PATH" ||
    $1 ~ /^NCCL_/ ||
    $1 ~ /^UCX_/ ||
    $1 == "FI_PROVIDER" ||
    $1 ~ /^FI_CXI_/ ||
    $1 ~ /^OMPI_/ ||
    $1 ~ /^PMI_/ ||
    $1 ~ /^PMIX_/ ||
    $1 ~ /^SLURM_(JOB_ID|JOB_NODELIST|NNODES|NTASKS|TASKS_PER_NODE|GPUS|GPUS_ON_NODE|GPUS_PER_TASK|LOCALID|PROCID)$/
'

section diagnostics
if ((${#diagnostics[@]} == 0)); then
    printf 'none\n'
else
    for diagnostic in "${diagnostics[@]}"; do
        printf 'partial: %s\n' "$diagnostic"
    done
fi

if ((${#diagnostics[@]} == 0)); then
    printf 'collector_status: complete\n'
    exit 0
fi

printf 'collector_status: partial\n'
printf 'next_action: %s\n' 'Run the collector in the target NVSHMEM environment on every participating compute node and retain the complete output.'
exit 2
