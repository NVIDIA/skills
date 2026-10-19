#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Collect read-only evidence used to select an NVSHMEM transport.
# Exit 0 means the core evidence categories were found; exit 2 means the
# output is still useful but one or more categories are incomplete.

set -u

usage() {
    printf '%s\n' \
        'Usage: collect-transport-facts.sh [--prefix PATH]' \
        '' \
        'Collect labeled, read-only NVSHMEM, GPU, RDMA, and provider facts.' \
        'Run this on the target compute node. No workload or benchmark is run.'
}

section() {
    printf '\n[%s]\n' "$1"
}

note_missing() {
    diagnostics+=("$1")
}

request_manual_follow_up() {
    manual_follow_ups+=("$1")
}

print_file() {
    local label=$1
    local path=$2
    if [[ -r "$path" ]]; then
        printf '%s: ' "$label"
        tr '\n' ' ' < "$path"
        printf '\n'
        return 0
    fi
    return 1
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

prefix_arg=''
while (($#)); do
    case "$1" in
        --prefix)
            if (($# < 2)); then
                printf 'error: --prefix requires PATH\n' >&2
                usage >&2
                exit 64
            fi
            prefix_arg=$2
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

diagnostics=()
manual_follow_ups=()
version_ok=0
plugins_ok=0
gpu_ok=0
fabric_ok=0
verbs_ok=0
mlx5_rdma_visible=0

nvshmem_prefix=''
prefix_source='unresolved'
if [[ -n "$prefix_arg" ]]; then
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

section collector
if ! hostname_value=$(hostname 2>/dev/null); then
    hostname_value='unknown'
fi
if ! effective_user=$(id -un 2>/dev/null); then
    effective_user="uid:$(id -u 2>/dev/null || printf unknown)"
fi
printf 'schema: nvshmem-transport-facts-v1\n'
printf 'timestamp_utc: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
printf 'hostname: %s\n' "$hostname_value"
printf 'effective_user: %s\n' "$effective_user"
printf 'prefix_source: %s\n' "$prefix_source"
if [[ -n "$nvshmem_prefix" ]]; then
    printf 'nvshmem_prefix: %s\n' "$nvshmem_prefix"
    if [[ ! -d "$nvshmem_prefix" ]]; then
        note_missing "resolved NVSHMEM prefix is not a readable directory: $nvshmem_prefix"
    fi
else
    printf 'nvshmem_prefix: unresolved\n'
    note_missing 'exact NVSHMEM installation prefix is unresolved'
fi

section nvshmem-version
info_bin=''
if [[ -n "$nvshmem_prefix" && -x "$nvshmem_prefix/bin/nvshmem-info" ]]; then
    info_bin="$nvshmem_prefix/bin/nvshmem-info"
elif command -v nvshmem-info >/dev/null 2>&1; then
    info_bin=$(command -v nvshmem-info)
fi

if [[ -n "$info_bin" ]]; then
    printf 'nvshmem_info: %s\n' "$info_bin"
    info_prefix_library_path=''
    if [[ -n "$nvshmem_prefix" ]]; then
        for libdir in "$nvshmem_prefix/lib" "$nvshmem_prefix/lib64"; do
            [[ -d "$libdir" ]] || continue
            if [[ -n "$info_prefix_library_path" ]]; then
                info_prefix_library_path+=":$libdir"
            else
                info_prefix_library_path=$libdir
            fi
        done
    fi

    if [[ -n "$info_prefix_library_path" ]]; then
        printf 'nvshmem_info_prefix_library_path: %s\n' "$info_prefix_library_path"
        info_ld_library_path=$info_prefix_library_path
        if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
            info_ld_library_path+=":$LD_LIBRARY_PATH"
        fi
        version_output=$(LC_ALL=C env LD_LIBRARY_PATH="$info_ld_library_path" "$info_bin" -n 2>&1)
    else
        version_output=$(LC_ALL=C "$info_bin" -n 2>&1)
    fi
    version_rc=$?
    printf '%s\n' "$version_output"
    printf 'nvshmem_info_exit_status: %d\n' "$version_rc"
    if ((version_rc == 0)) && grep -Eq '[0-9]+\.[0-9]+([.][0-9]+)?' <<< "$version_output"; then
        version_ok=1
    fi
else
    printf 'nvshmem_info: not found\n'
fi

if ((version_ok == 0)) && [[ -n "$nvshmem_prefix" ]]; then
    version_file="$nvshmem_prefix/version.txt"
    if [[ -r "$version_file" ]]; then
        version_file_evidence=$(grep -E '^NVSHMEM_(BASE|ARTIFACT|PACKAGE)_VERSION[[:space:]]*:=[[:space:]]*[0-9]+\.[0-9]+([.][0-9]+)?([^[:space:]]*)?[[:space:]]*$' "$version_file" 2>/dev/null || true)
        if [[ -n "$version_file_evidence" ]]; then
            printf 'version_file: %s\n%s\n' "$version_file" "$version_file_evidence"
            version_ok=1
        fi
    fi
fi

if ((version_ok == 0)) && [[ -n "$nvshmem_prefix" ]]; then
    version_header="$nvshmem_prefix/include/non_abi/nvshmem_version.h"
    if [[ -r "$version_header" ]]; then
        version_header_evidence=$(grep -E '^#[[:space:]]*define[[:space:]]+NVSHMEM_VENDOR_(BASE|ARTIFACT|PACKAGE)_VERSION_STRING[[:space:]]+"[0-9]+\.[0-9]+([.][0-9]+)?[^"]*"' "$version_header" 2>/dev/null || true)
        if [[ -n "$version_header_evidence" ]]; then
            printf 'header: %s\n%s\n' "$version_header" "$version_header_evidence"
            version_ok=1
        fi
    fi
fi

if ((version_ok == 0)) && [[ -n "$nvshmem_prefix" ]]; then
    for version_header in \
        "$nvshmem_prefix/include/non_abi/nvshmem_version.h" \
        "$nvshmem_prefix/include/device_host_transport/nvshmem_constants.h"; do
        if [[ -r "$version_header" ]]; then
            matches=$(grep -E '^#[[:space:]]*define[[:space:]]+NVSHMEM_VENDOR_((MAJOR|MINOR|PATCH)_VERSION[[:space:]]+[0-9]+|VERSION_STAGE(_NUMBER)?[[:space:]]+"[^"]*")([[:space:]]|$)' "$version_header" 2>/dev/null || true)
            if [[ -n "$matches" ]]; then
                printf 'header: %s\n%s\n' "$version_header" "$matches"
            fi
            if grep -Eq 'NVSHMEM_VENDOR_MAJOR_VERSION[[:space:]]+[0-9]+([[:space:]]|$)' <<< "$matches" &&
                grep -Eq 'NVSHMEM_VENDOR_MINOR_VERSION[[:space:]]+[0-9]+([[:space:]]|$)' <<< "$matches" &&
                grep -Eq 'NVSHMEM_VENDOR_PATCH_VERSION[[:space:]]+[0-9]+([[:space:]]|$)' <<< "$matches"; then
                version_ok=1
                break
            fi
        fi
    done
fi
if ((version_ok == 0)); then
    note_missing 'exact NVSHMEM version was not found'
fi

section nvshmem-plugins
plugin_count=0
plugin_file_count=0
declare -A plugin_transports_seen=()
if [[ -n "$nvshmem_prefix" && -d "$nvshmem_prefix" ]]; then
    shopt -s nullglob
    for libdir in "$nvshmem_prefix/lib" "$nvshmem_prefix/lib64"; do
        [[ -d "$libdir" ]] || continue
        printf 'searched_directory: %s\n' "$libdir"
        plugin_files=(
            "$libdir"/nvshmem_transport_*.so*
            "$libdir"/libnvshmem_transport_*.so*
        )
        for plugin in "${plugin_files[@]}"; do
            [[ -e "$plugin" || -L "$plugin" ]] || continue
            printf 'plugin: %s\n' "$plugin"
            ((plugin_file_count += 1))
            plugin_filename=${plugin##*/}
            plugin_transport=${plugin_filename#libnvshmem_transport_}
            plugin_transport=${plugin_transport#nvshmem_transport_}
            plugin_transport=${plugin_transport%%.so*}
            if [[ -n "$plugin_transport" ]]; then
                plugin_transports_seen["$plugin_transport"]=1
            fi
        done
    done
    shopt -u nullglob
    plugins_ok=1
    plugin_count=${#plugin_transports_seen[@]}
    if ((plugin_count)); then
        while IFS= read -r plugin_transport; do
            printf 'plugin_transport: %s\n' "$plugin_transport"
        done < <(printf '%s\n' "${!plugin_transports_seen[@]}" | LC_ALL=C sort)
    fi
    printf 'plugin_count_in_selected_prefix: %d\n' "$plugin_count"
    printf 'plugin_file_count_in_selected_prefix: %d\n' "$plugin_file_count"
else
    printf 'selected_prefix_inventory: unavailable\n'
    if command -v ldconfig >/dev/null 2>&1; then
        printf 'loader_cache_hints (not an exact-prefix inventory):\n'
        ldconfig -p 2>/dev/null | grep -E 'nvshmem.*transport|transport.*nvshmem' || printf 'none\n'
    fi
fi
if ((plugins_ok == 0)); then
    note_missing 'selected-prefix transport plugin inventory is unavailable'
elif ((plugin_count == 0)); then
    printf 'plugin_inventory_result: empty\n'
fi

section nvshmem-configuration
printf 'transport_related_environment:\n'
selection_environment=$(env | LC_ALL=C grep -E '^(NVSHMEM_(REMOTE_TRANSPORT|IB_ENABLE_IBGDA|GPUNETIO_ENABLE_GDAKI|LIBFABRIC_PROVIDER|BOOTSTRAP|SYMMETRIC_SIZE|DISABLE_CUDA_VMM|ENABLE_NIC_PE_MAPPING|HCA_LIST|HCA_PE_MAPPING)|UCX_TLS|FI_PROVIDER|FI_EFA_ENABLE_SHM_TRANSFER)=' | LC_ALL=C sort || true)
if [[ -n "$selection_environment" ]]; then
    printf '%s\n' "$selection_environment"
else
    printf 'none\n'
fi
for config_path in \
    "${NVSHMEM_CONFIG_FILE:-}" \
    "${nvshmem_prefix:+$nvshmem_prefix/etc/nvshmem.conf}"; do
    [[ -n "$config_path" ]] || continue
    if [[ -r "$config_path" ]]; then
        printf 'configuration_file: %s\n' "$config_path"
        grep -E '^[[:space:]]*(NVSHMEM_)?(REMOTE_TRANSPORT|IB_ENABLE_IBGDA|GPUNETIO_ENABLE_GDAKI|LIBFABRIC_PROVIDER|DISABLE_CUDA_VMM|HCA_LIST|HCA_PE_MAPPING)[[:space:]]*=' "$config_path" 2>/dev/null || printf 'no transport-selection entries\n'
    fi
done

section gpu
if command -v nvidia-smi >/dev/null 2>&1; then
    if run_probe gpu_inventory nvidia-smi --query-gpu=index,name,pci.bus_id,compute_cap,driver_version --format=csv,noheader; then
        gpu_ok=1
    fi
    run_probe gpu_topology nvidia-smi topo -m || true
    run_probe gpu_p2p_read nvidia-smi topo -p2p r || true
else
    printf 'nvidia_smi: not found\n'
    note_missing 'GPU and driver evidence is unavailable'
fi
if ((gpu_ok == 0)) && command -v nvidia-smi >/dev/null 2>&1; then
    note_missing 'nvidia-smi did not return a GPU inventory'
fi
if ((gpu_ok == 0)); then
    request_manual_follow_up 'nvidia-smi --query-gpu=index,name,pci.bus_id,compute_cap,driver_version --format=csv,noheader'
    request_manual_follow_up 'nvidia-smi topo -m'
fi

section kernel-driver
run_probe uname uname -a || true
print_file nvidia_driver_version /proc/driver/nvidia/version || printf 'nvidia_driver_version: unavailable\n'
if [[ -r /proc/driver/nvidia/params ]]; then
    printf 'PeerMappingOverride: '
    grep -E '^PeerMappingOverride:' /proc/driver/nvidia/params 2>/dev/null || printf 'not present\n'
else
    printf 'PeerMappingOverride: unavailable\n'
fi
for module in nvidia nvidia_peermem nv_peer_mem; do
    if [[ -d "/sys/module/$module" ]]; then
        printf 'module: %s loaded\n' "$module"
        print_file "${module}_version" "/sys/module/$module/version" || true
        print_file "${module}_initstate" "/sys/module/$module/initstate" || true
    else
        printf 'module: %s not_loaded\n' "$module"
    fi
done
if [[ -d /sys/module/dma_buf ]]; then
    printf 'dma_buf_kernel_module: present\n'
else
    printf 'dma_buf_kernel_module: not_visible\n'
fi
if [[ -d /sys/kernel/dmabuf ]]; then
    printf 'dma_buf_sysfs: present\n'
else
    printf 'dma_buf_sysfs: not_visible\n'
fi
kernel_release=$(uname -r 2>/dev/null || true)
for kernel_config in /proc/config.gz "/boot/config-$kernel_release"; do
    [[ -r "$kernel_config" ]] || continue
    printf 'dma_buf_kernel_config: %s\n' "$kernel_config"
    if [[ "$kernel_config" == *.gz ]] && command -v zgrep >/dev/null 2>&1; then
        zgrep -E '^CONFIG_(DMA_SHARED_BUFFER|DMABUF_MOVE_NOTIFY)=' "$kernel_config" 2>/dev/null || printf 'not reported\n'
    else
        grep -E '^CONFIG_(DMA_SHARED_BUFFER|DMABUF_MOVE_NOTIFY)=' "$kernel_config" 2>/dev/null || printf 'not reported\n'
    fi
done
printf 'dma_buf_note: kernel infrastructure alone does not prove a working GPU RDMA registration path\n'
if command -v modinfo >/dev/null 2>&1; then
    for module in nvidia_peermem nv_peer_mem; do
        printf '%s_modinfo:\n' "$module"
        modinfo -F version "$module" 2>&1 || true
    done
fi

section rdma-ports
shopt -s nullglob
ib_devices=(/sys/class/infiniband/*)
if ((${#ib_devices[@]})); then
    fabric_ok=1
    for ibdev_path in "${ib_devices[@]}"; do
        ibdev=${ibdev_path##*/}
        printf 'rdma_device: %s\n' "$ibdev"
        canonical_device=$(readlink -f "$ibdev_path/device" 2>/dev/null || true)
        [[ -n "$canonical_device" ]] && printf '  pci_device: %s\n' "${canonical_device##*/}"
        print_file '  vendor' "$ibdev_path/device/vendor" || true
        print_file '  device' "$ibdev_path/device/device" || true
        driver_link=$(readlink -f "$ibdev_path/device/driver" 2>/dev/null || true)
        if [[ -n "$driver_link" ]]; then
            driver_name=${driver_link##*/}
            printf '  driver: %s\n' "$driver_name"
            [[ "$driver_name" == mlx5_core ]] && mlx5_rdma_visible=1
        fi
        net_paths=("$ibdev_path"/device/net/*)
        for net_path in "${net_paths[@]}"; do
            [[ -e "$net_path" ]] && printf '  netdev: %s\n' "${net_path##*/}"
        done
        port_paths=("$ibdev_path"/ports/*)
        for port_path in "${port_paths[@]}"; do
            [[ -d "$port_path" ]] || continue
            port=${port_path##*/}
            printf '  port: %s\n' "$port"
            print_file '    state' "$port_path/state" || true
            print_file '    phys_state' "$port_path/phys_state" || true
            print_file '    link_layer' "$port_path/link_layer" || true
            print_file '    rate' "$port_path/rate" || true
        done
    done
else
    printf 'rdma_device: none visible in /sys/class/infiniband\n'
fi
shopt -u nullglob

section rdma-stack
if command -v ofed_info >/dev/null 2>&1; then
    run_probe ofed_version ofed_info -s || true
else
    printf 'ofed_info: not found\n'
fi
if command -v rdma >/dev/null 2>&1; then
    run_probe rdma_version rdma -V || true
    if ! run_probe rdma_links rdma link show; then
        request_manual_follow_up 'rdma link show'
    fi
else
    printf 'rdma: not found\n'
fi
if command -v ibv_devinfo >/dev/null 2>&1; then
    printf 'verbs_devices:\n'
    verbs_output=$(LC_ALL=C ibv_devinfo -l 2>&1)
    verbs_rc=$?
    printf '%s\n' "$verbs_output"
    printf 'verbs_devices_exit_status: %d\n' "$verbs_rc"
    if ((verbs_rc == 0)) && grep -Eq '^[[:space:]]*[1-9][0-9]* HCAs? found:' <<< "$verbs_output"; then
        verbs_ok=1
    fi
else
    printf 'ibv_devinfo: not found\n'
fi
if ((mlx5_rdma_visible && verbs_ok == 0)); then
    note_missing 'mlx5 RDMA hardware is visible in sysfs but no verbs HCA is accessible in this execution context'
    request_manual_follow_up 'ibstatus'
    request_manual_follow_up 'ibv_devinfo -l'
fi

section gdrcopy
gdrcopy_driver_evidence=0
if [[ -e /dev/gdrdrv ]]; then
    printf 'gdrcopy_device: /dev/gdrdrv present\n'
    gdrcopy_driver_evidence=1
else
    printf 'gdrcopy_device: /dev/gdrdrv not_visible\n'
fi
if [[ -d /sys/module/gdrdrv ]]; then
    printf 'gdrcopy_module: gdrdrv loaded\n'
    print_file gdrcopy_module_version /sys/module/gdrdrv/version || true
    print_file gdrcopy_module_initstate /sys/module/gdrdrv/initstate || true
    gdrcopy_driver_evidence=1
else
    printf 'gdrcopy_module: gdrdrv not_loaded\n'
fi

if ((gdrcopy_driver_evidence)); then
    printf 'gdrcopy_status: driver_present\n'
else
    printf 'gdrcopy_status: driver_not_detected\n'
fi
printf 'gdrcopy_note: userspace library hints are reported under provider-stacks; their absence does not disprove a custom installation\n'

section provider-stacks
if command -v ucx_info >/dev/null 2>&1; then
    run_probe ucx_version ucx_info -v || true
    printf 'ucx_transports:\n'
    LC_ALL=C ucx_info -d 2>&1 | LC_ALL=C grep -E '^#.*(Transport|Device):' || printf 'not reported\n'
else
    printf 'ucx_info: not found\n'
fi
if command -v fi_info >/dev/null 2>&1; then
    run_probe libfabric_version fi_info --version || true
    if run_probe libfabric_providers fi_info -l; then
        fabric_ok=1
    fi
else
    printf 'fi_info: not found\n'
fi
if command -v pkg-config >/dev/null 2>&1; then
    for package in libfabric ucx libmlx5 gdrapi gpunetio doca-gpunetio doca; do
        if pkg-config --exists "$package" 2>/dev/null; then
            printf 'pkg_config: %s %s\n' "$package" "$(pkg-config --modversion "$package" 2>/dev/null || printf unknown)"
        else
            printf 'pkg_config: %s not_found\n' "$package"
        fi
    done
else
    printf 'pkg-config: not found\n'
fi
if command -v ldconfig >/dev/null 2>&1; then
    printf 'provider_library_hints:\n'
    ldconfig -p 2>/dev/null | grep -E 'lib(ucp|ucs|uct|fabric|mlx5|efa|cxi|gdrapi|gpunetio|doca)' || printf 'none\n'
fi

section diagnostics
if ((fabric_ok == 0)); then
    note_missing 'neither an RDMA device nor a libfabric provider inventory was available'
fi
if ((${#diagnostics[@]})); then
    for diagnostic in "${diagnostics[@]}"; do
        printf 'partial: %s\n' "$diagnostic"
    done
else
    printf 'none\n'
fi
if ((${#manual_follow_ups[@]})); then
    printf 'manual_follow_up_context: run these commands manually in the target host or exact launch environment and return their complete output\n'
    for follow_up_command in "${manual_follow_ups[@]}"; do
        printf 'manual_follow_up_command: %s\n' "$follow_up_command"
    done
fi

if ((version_ok && plugins_ok && gpu_ok && fabric_ok)) &&
    ((${#diagnostics[@]} == 0 && ${#manual_follow_ups[@]} == 0)); then
    printf 'collector_status: complete\n'
    exit 0
fi

printf 'collector_status: partial\n'
exit 2
