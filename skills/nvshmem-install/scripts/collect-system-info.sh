#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Read-only, unprivileged NVSHMEM installation preflight.
# Do not add sudo, package installation, network access, builds, or filesystem writes.

set -uo pipefail

emit() {
    local key="$1"
    local value="${2:-unknown}"
    value="$(printf '%s' "$value" | tr '\n\t' '  ' | sed 's/[[:space:]][[:space:]]*/ /g; s/^ //; s/ $//')"
    if [[ -z "$value" ]]; then
        value="unknown"
    fi
    printf '%s=%s\n' "$key" "$value"
}

first_line() {
    "$@" 2>/dev/null | sed -n '1p'
}

last_line() {
    "$@" 2>/dev/null | sed -n '$p'
}

command_path() {
    command -v "$1" 2>/dev/null || true
}

command_list() {
    local result=""
    local command_name
    for command_name in "$@"; do
        if command -v "$command_name" >/dev/null 2>&1; then
            result="${result}${result:+,}${command_name}"
        fi
    done
    printf '%s' "${result:-none}"
}

os_value() {
    local key="$1"
    if [[ -r /etc/os-release ]]; then
        sed -n "s/^${key}=//p" /etc/os-release | sed -n '1p' | tr -d '"'
    fi
}

emit probe_version "1"
emit hostname "$(hostname 2>/dev/null || true)"
emit kernel "$(uname -sr 2>/dev/null || true)"
emit cpu_arch "$(uname -m 2>/dev/null || true)"
emit cpu_model "$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | sed -n '1p')"
emit os_id "$(os_value ID)"
emit os_version "$(os_value VERSION_ID)"

container_type="none"
if [[ -f /.dockerenv ]]; then
    container_type="docker-compatible"
elif [[ -r /proc/1/cgroup ]] && grep -Eq 'docker|containerd|kubepods|libpod' /proc/1/cgroup 2>/dev/null; then
    container_type="container-cgroup"
elif [[ -n "${APPTAINER_NAME:-}" || -n "${SINGULARITY_NAME:-}" ]]; then
    container_type="apptainer-or-singularity"
fi
emit running_in_container "$container_type"
emit container_runtimes "$(command_list docker podman apptainer singularity)"

if command -v nvidia-smi >/dev/null 2>&1; then
    emit nvidia_smi "$(command_path nvidia-smi)"
    if gpu_rows="$(nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv,noheader 2>/dev/null)"; then
        emit gpu_count "$(awk 'NF { count++ } END { print count + 0 }' <<<"$gpu_rows")"
        emit gpu_inventory "$gpu_rows"
        emit gpu_topology "$(nvidia-smi topo -m 2>/dev/null || true)"
    else
        emit gpu_count "unknown"
        emit gpu_inventory "nvidia-smi-present-but-unavailable"
        emit gpu_topology "unknown"
    fi
else
    emit nvidia_smi "missing"
    emit gpu_count "unknown"
    emit gpu_inventory "unknown"
    emit gpu_topology "unknown"
fi

emit nvidia_driver_module "$(first_line cat /proc/driver/nvidia/version)"
emit nvidia_peermem_module "$([[ -d /sys/module/nvidia_peermem ]] && printf 'loaded' || printf 'not-detected')"
emit cuda_home "${CUDA_HOME:-unknown}"
emit nvcc "$(command_path nvcc)"
emit nvcc_version "$(last_line nvcc --version)"
emit cmake "$(command_path cmake)"
emit cmake_version "$(first_line cmake --version)"

cxx_command=""
for candidate in c++ g++ clang++; do
    if command -v "$candidate" >/dev/null 2>&1; then
        cxx_command="$candidate"
        break
    fi
done
emit cxx_compiler "${cxx_command:-missing}"
if [[ -n "$cxx_command" ]]; then
    emit cxx_version "$(first_line "$cxx_command" --version)"
else
    emit cxx_version "unknown"
fi

emit python "$(command_path python3)"
emit python_version "$(first_line python3 --version)"
emit python_package_tools "$(command_list pip pip3 conda mamba virtualenv)"
emit package_managers "$(command_list apt apt-get dpkg dnf yum rpm zypper)"

emit slurm_job_id "${SLURM_JOB_ID:-none}"
emit slurm_nodes "${SLURM_JOB_NODELIST:-none}"
emit launchers "$(command_list srun mpirun mpiexec nvshmrun oshrun)"
emit srun_version "$(first_line srun --version)"
emit mpi_version "$(first_line mpirun --version)"
emit pmix_info "$(command_path pmix_info)"

emit rdma_tools "$(command_list ibv_devices ibv_devinfo rdma)"
emit rdma_devices "$(ibv_devices 2>/dev/null || true)"
emit rdma_link_details "$(rdma link show 2>/dev/null || true)"
emit rdma_device_details "$(ibv_devinfo 2>/dev/null || true)"
emit ofed_version "$(ofed_info -s 2>/dev/null | sed -n '1p')"
emit ucx_version "$(ucx_info -v 2>/dev/null | sed -n '1p')"
emit libfabric_version "$(fi_info --version 2>/dev/null | sed -n '1p')"
emit libfabric_providers "$(fi_info -l 2>/dev/null || true)"
emit doca_info "$(command_path doca_info)"
emit doca_version "$(first_line doca_info --version)"
emit doca_sdk_lib_path "${DOCA_SDK_LIB_PATH:-unknown}"

emit nvshmem_home "${NVSHMEM_HOME:-unknown}"
emit nvshmem_prefix "${NVSHMEM_PREFIX:-unknown}"
