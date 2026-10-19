/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nvshmem.h>

__global__ void exchange_flags(unsigned long long *remote_flags, int npes) {
    const int lane = threadIdx.x & 31;
    const int peer = (blockIdx.x + (threadIdx.x >> 5) + 1) % npes;
    if (lane == 0) {
        for (int round = 0; round < 4096; ++round) {
            nvshmem_ulonglong_p(remote_flags + blockIdx.x, round, peer);
        }
    }
    nvshmem_quiet();
}

// Launch: 128 CTAs x 256 threads. Each warp leader chooses a peer.
