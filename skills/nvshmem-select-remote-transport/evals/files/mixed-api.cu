/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

__global__ void persistent_exchange(unsigned long long *remote, int npes) {
    int peer = (blockIdx.x + 1) % npes;
    for (int i = threadIdx.x; i < 8192; i += blockDim.x) {
        nvshmem_ulonglong_p(remote + i, i, peer);
    }
    nvshmem_quiet();
}

void publish_control(int *remote_control, const int *local, int peer, cudaStream_t stream) {
    nvshmemx_int_put_on_stream(remote_control, local, 1, peer, stream);
}
