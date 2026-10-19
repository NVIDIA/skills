#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Read-only NVSHMEM GPU/NIC topology collector. All evidence is emitted to stdout.
set -u
exec 2>&1

usage() {
    printf '%s\n' \
        'Usage: collect-nic-topology.sh [--prefix PATH]' \
        '' \
        'Collect labeled, read-only NVSHMEM version, GPU, and NIC topology evidence.' \
        'Run this on the target compute node. No workload or benchmark is run.'
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

critical_failures=0
version_ok=0
diagnostics=()

add_diagnostic() {
    diagnostics+=("$1")
}

print_command_result() {
    local command_label=$1
    local command_output=$2
    local command_status=$3

    printf 'command=%q exit_status=%d\n' "$command_label" "$command_status"
    if [[ -n "$command_output" ]]; then
        printf '%s\n' "$command_output"
    else
        printf 'output=(empty)\n'
    fi
}

nvshmem_prefix=''
prefix_source='unresolved'
if ((prefix_arg_set)); then
    nvshmem_prefix=$prefix_arg
    prefix_source='--prefix'
elif [[ -n "${NVSHMEM_PREFIX:-}" ]]; then
    nvshmem_prefix=$NVSHMEM_PREFIX
    prefix_source='NVSHMEM_PREFIX'
elif command -v nvshmem-info >/dev/null 2>&1; then
    info_on_path=$(command -v nvshmem-info)
    nvshmem_prefix=$(dirname "$(dirname "$info_on_path")")
    prefix_source='nvshmem-info on PATH'
fi

printf '[collector]\n'
printf 'prefix_source=%q\n' "$prefix_source"
if [[ -n "$nvshmem_prefix" ]]; then
    printf 'nvshmem_prefix=%q\n' "$nvshmem_prefix"
    if [[ ! -d "$nvshmem_prefix" || ! -r "$nvshmem_prefix" ]]; then
        critical_failures=1
        add_diagnostic "resolved NVSHMEM prefix is not a readable directory: $nvshmem_prefix"
    fi
else
    printf 'nvshmem_prefix=unresolved\n'
fi

printf '[nvshmem-version]\n'
info_bin=''
info_source='unavailable'
if [[ -n "$nvshmem_prefix" && -f "$nvshmem_prefix/bin/nvshmem-info" &&
    -x "$nvshmem_prefix/bin/nvshmem-info" ]]; then
    info_bin="$nvshmem_prefix/bin/nvshmem-info"
    info_source='resolved prefix'
elif command -v nvshmem-info >/dev/null 2>&1; then
    info_bin=$(command -v nvshmem-info)
    info_source='PATH'
fi

if [[ -n "$info_bin" ]]; then
    printf 'nvshmem_info=%q\n' "$info_bin"
    printf 'nvshmem_info_source=%q\n' "$info_source"

    info_prefix_library_path=''
    if [[ "$info_source" == 'resolved prefix' ]]; then
        for libdir in "$nvshmem_prefix/lib" "$nvshmem_prefix/lib64"; do
            [[ -d "$libdir" ]] || continue
            if [[ -n "$info_prefix_library_path" ]]; then
                info_prefix_library_path+=":$libdir"
            else
                info_prefix_library_path=$libdir
            fi
        done
    fi
    printf 'nvshmem_info_prefix_library_path=%q\n' "${info_prefix_library_path:-none}"

    if [[ -n "$info_prefix_library_path" ]]; then
        info_ld_library_path=$info_prefix_library_path
        if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
            info_ld_library_path+=":$LD_LIBRARY_PATH"
        fi
        nvshmem_output="$(LC_ALL=C env LD_LIBRARY_PATH="$info_ld_library_path" "$info_bin" -n 2>&1)"
    else
        nvshmem_output="$(LC_ALL=C "$info_bin" -n 2>&1)"
    fi
    nvshmem_status=$?
    print_command_result 'nvshmem-info -n' "$nvshmem_output" "$nvshmem_status"
    if ((nvshmem_status != 0)); then
        critical_failures=1
        add_diagnostic 'nvshmem-info -n failed; provide the installed NVSHMEM version explicitly.'
    elif [[ -z "$nvshmem_output" ]]; then
        critical_failures=1
        add_diagnostic 'nvshmem-info -n returned no version text; provide the installed NVSHMEM version explicitly.'
    elif ! grep -Eq '[0-9]+\.[0-9]+([.][0-9]+)?' <<< "$nvshmem_output"; then
        critical_failures=1
        add_diagnostic 'nvshmem-info -n did not report a recognizable NVSHMEM version.'
    else
        version_ok=1
    fi
else
    printf 'command=%q status=unavailable\n' 'nvshmem-info -n'
    critical_failures=1
    add_diagnostic 'nvshmem-info is unavailable; provide --prefix, set NVSHMEM_PREFIX, or supply the installed version explicitly.'
fi

printf '[gpu-pci]\n'
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_output="$(nvidia-smi --query-gpu=index,pci.bus_id,name --format=csv,noheader 2>&1)"
    gpu_status=$?
    print_command_result 'nvidia-smi --query-gpu=index,pci.bus_id,name --format=csv,noheader' "$gpu_output" "$gpu_status"
    if ((gpu_status != 0)); then
        critical_failures=1
        add_diagnostic 'GPU PCI discovery failed; run this collector on an allocated compute node with a working NVIDIA driver.'
    fi
else
    printf 'command=%q status=unavailable\n' 'nvidia-smi --query-gpu=index,pci.bus_id,name --format=csv,noheader'
    critical_failures=1
    add_diagnostic 'nvidia-smi is not on PATH; run this collector on a GPU compute node.'
fi

printf '[gpu-nic-topology]\n'
if command -v nvidia-smi >/dev/null 2>&1; then
    topo_output="$(nvidia-smi topo -m 2>&1)"
    topo_status=$?
    print_command_result 'nvidia-smi topo -m' "$topo_output" "$topo_status"
    if ((topo_status != 0)); then
        critical_failures=1
        add_diagnostic 'GPU/NIC topology discovery failed; run this collector on the target compute node, not a login node.'
    fi
else
    printf 'command=%q status=unavailable\n' 'nvidia-smi topo -m'
fi

printf '[rdma-ports]\n'
shopt -s nullglob
port_paths=(/sys/class/infiniband/*/ports/*)
ports_found=0

for port_path in "${port_paths[@]}"; do
    [[ -d "$port_path" ]] || continue
    ports_found=$((ports_found + 1))

    hca_dir=${port_path%/ports/*}
    hca_name=${hca_dir##*/}
    port_number=${port_path##*/}

    state='unknown'
    if [[ -r "$port_path/state" ]]; then
        state="$(<"$port_path/state")"
    fi

    link_layer='unknown'
    if [[ -r "$port_path/link_layer" ]]; then
        link_layer="$(<"$port_path/link_layer")"
    fi

    active='no'
    if [[ "$state" == *ACTIVE* ]]; then
        active='yes'
    fi

    pci_device='unknown'
    device_path="$(readlink -f "$hca_dir/device" 2>/dev/null)"
    if [[ -n "$device_path" ]]; then
        pci_device=${device_path##*/}
    fi

    net_names=()
    net_paths=("$hca_dir"/device/net/*)
    for net_path in "${net_paths[@]}"; do
        net_names+=("${net_path##*/}")
    done
    netdev='none'
    if ((${#net_names[@]} > 0)); then
        netdev="$(IFS=,; printf '%s' "${net_names[*]}")"
    fi

    printf 'hca=%q port=%q pci=%q state=%q active=%q link_layer=%q netdev=%q\n' \
        "$hca_name" "$port_number" "$pci_device" "$state" "$active" "$link_layer" "$netdev"
done

if ((ports_found == 0)); then
    printf 'ports_found=0\n'
    add_diagnostic 'No HCA ports were readable under /sys/class/infiniband; checking ibv_devinfo as a fallback.'

    printf '[ibv-devinfo-fallback]\n'
    if command -v ibv_devinfo >/dev/null 2>&1; then
        ibv_output="$(ibv_devinfo -v 2>&1)"
        ibv_status=$?
        print_command_result 'ibv_devinfo -v' "$ibv_output" "$ibv_status"
        if ((ibv_status != 0)); then
            add_diagnostic 'ibv_devinfo also failed; verify RDMA device visibility and permissions on the target node.'
        fi
    else
        printf 'command=%q status=unavailable\n' 'ibv_devinfo -v'
        add_diagnostic 'ibv_devinfo is not on PATH; install or load RDMA utilities, or provide per-port state and link-layer evidence.'
    fi

    critical_failures=1
fi

printf '[diagnostics]\n'
if ((${#diagnostics[@]} == 0)); then
    printf 'message=none\n'
else
    for diagnostic in "${diagnostics[@]}"; do
        printf 'message=%q\n' "$diagnostic"
    done
fi

if ((critical_failures == 0 && version_ok)); then
    printf 'status=complete\n'
    exit 0
fi

printf 'status=partial\n'
printf 'next_action=%q\n' 'Run on the target allocated compute node and paste the complete labeled output together with local PEs per node and the local PE-to-GPU binding.'
exit 2
