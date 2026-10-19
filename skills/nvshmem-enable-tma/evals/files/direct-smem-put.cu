/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

__global__ void produce_and_send(float *remote_dst, float *staging, size_t nelems, int peer) {
    extern __shared__ float tile[];

    for (size_t i = threadIdx.x; i < nelems; i += blockDim.x) {
        tile[i] = static_cast<float>(i);
    }
    __syncthreads();

    for (size_t i = threadIdx.x; i < nelems; i += blockDim.x) {
        staging[i] = tile[i];
    }
    __syncthreads();

    nvshmemx_float_put_nbi_block(remote_dst, staging, nelems, peer);
    nvshmem_quiet();
}

void launch(float *remote_dst, float *staging, size_t nelems, int peer, cudaStream_t stream) {
    size_t tile_bytes = nelems * sizeof(float);
    produce_and_send<<<1, 256, tile_bytes, stream>>>(remote_dst, staging, nelems, peer);
}
