/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nvshmem.h>

__global__ void send_packed(char *remote_dst, const char *packed, int peer) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        nvshmem_putmem_nbi(remote_dst, packed, 65536, peer);
        nvshmem_quiet();
    }
}

// One issuing lane; one 64 KiB physical transfer per iteration.
