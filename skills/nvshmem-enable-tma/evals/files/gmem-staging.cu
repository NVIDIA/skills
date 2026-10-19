/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

__global__ void communicate(float *remote_dst, const float *local_src, size_t nelems, int peer) {
    nvshmemx_float_put_nbi_block(remote_dst, local_src, nelems, peer);
    nvshmem_quiet();
}

__global__ void unrelated_scale(float *data, size_t nelems) {
    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < nelems;
         i += blockDim.x * gridDim.x) {
        data[i] *= 2.0f;
    }
}

void launch(float *remote_dst, const float *local_src, float *local_data, size_t nelems, int peer,
            cudaStream_t stream) {
    communicate<<<1, 256, 0, stream>>>(remote_dst, local_src, nelems, peer);
    unrelated_scale<<<16, 256, 0, stream>>>(local_data, nelems);
}
