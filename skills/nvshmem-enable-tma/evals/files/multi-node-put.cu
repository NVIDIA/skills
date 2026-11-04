/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

__global__ void communicate(int *remote_dst, const int *local_src, size_t nelems, int peer) {
    nvshmemx_int_put_nbi_block(remote_dst, local_src, nelems, peer);
    nvshmem_quiet();
}

void launch(int *remote_dst, const int *local_src, size_t nelems, int peer) {
    communicate<<<1, 256>>>(remote_dst, local_src, nelems, peer);
}
