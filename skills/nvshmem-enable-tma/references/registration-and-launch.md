# TMA Shared-Memory Registration and Launch

Read this reference completely for every TMA code change. Read the selected path reference as well.

## Canonical NVSHMEM API (3.7.X and later)

Use only the symbols below. They are declared by `<nvshmemx.h>`; do not redeclare them, invent aliases, or substitute CUDA TMA APIs. If the target installation does not declare them, stop and report that its headers do not match the requested 3.7.0-or-newer TMA API.

The NVSHMEM 3.7.0, 3.7.1, and 3.7.2 TMA documentation contains an incorrect single-thread requirement for `nvshmemx_give_smem` and `nvshmemx_release_smem`. Both calls are CTA-wide: every thread in each participating CTA must call both functions on the same CTA-uniform path. Every thread must pass the same shared-memory pointer and size to `nvshmemx_give_smem`. Never guard either call with `threadIdx.x == 0`, an elected lane, or another leader-only condition. This requirement overrides the archived 3.7.0 through 3.7.2 wording.

Pinned NVIDIA sources:

- [Using TMA with NVSHMEM — NVSHMEM 3.7.0](https://archive.docs.nvidia.com/nvshmem/api/3.7.0/tma.html)
- [NVSHMEM 3.7.0 release notes](https://archive.docs.nvidia.com/nvshmem/3.7.0/release-notes-install-guide/release-notes/release-3700.html)

```cuda
typedef enum {
    NVSHMEMX_SMEM_RECOMMENDED,
    NVSHMEMX_SMEM_MINIMUM,
    NVSHMEMX_SMEM_BARRIERS_ONLY
} nvshmemx_smem_amount_t;

__host__ __device__ int nvshmemx_ask_smem(
    nvshmemx_smem_amount_t flag);
__device__ void nvshmemx_give_smem(void *smem, size_t size);
__device__ void nvshmemx_release_smem(void);
```

- Treat `NVSHMEM_TMA_POLICY` as an environment variable, not a C/C++ symbol. Set `NVSHMEM_TMA_POLICY=ENABLE` before any `nvshmem_init*` call.
- Keep the existing NVSHMEM put/get API. NVSHMEM exposes no TMA-specific RMA API.

## Select the SMEM Amount

Select the `nvshmemx_ask_smem` flag from the selected operations' data path:

- Use `NVSHMEMX_SMEM_RECOMMENDED` when any selected operation may require NVSHMEM GMEM staging.
- Use `NVSHMEMX_SMEM_BARRIERS_ONLY` only when every selected operation is a direct application-owned shared-memory put source and needs only NVSHMEM's internal TMA state.
- Use `NVSHMEMX_SMEM_MINIMUM` only after the user explicitly accepts its occupancy/capability tradeoff and the inspected operations support the smaller allocation.

Query the selected amount on the host, pass the returned size to the kernel, and keep it identical across every participating CTA in a launch. Never hard-code a byte count from a particular release.

## Mandatory Lifetime Protocol

Reject an incomplete patch. Every participating CTA must follow this protocol on a CTA-uniform path:

- [ ] Reserve a 16-byte-aligned prefix at least as large as the queried amount.
- [ ] Have every CTA thread call `nvshmemx_give_smem(prefix, prefix_size)` once per registration lifetime on the same CTA-uniform path. All threads must pass the same pointer and size. Then execute `__syncthreads()` before any CTA thread enters a TMA-eligible operation.
- [ ] Drain work before release: use the scope-matched `nvshmemx_flush`, `nvshmemx_flush_warp`, or `nvshmemx_flush_block` only when source reuse is sufficient; use `nvshmem_quiet` for remote completion or full ordering. `nvshmem_fence` is not a completion drain.
- [ ] Execute `__syncthreads()` after all issuing threads reach the drain and before release.
- [ ] Have every CTA thread call `nvshmemx_release_smem()` on the same CTA-uniform path before the CTA returns. No thread that calls `give_smem` may bypass its matching release. Synchronize again before reusing the prefix after release.

Use this device-side shape, adapting only the RMA call and completion strength:

```cuda
nvshmemx_give_smem(nvshmem_prefix, nvshmem_prefix_size);
__syncthreads();

// Existing NVSHMEM put/get/put-with-signal call(s).
nvshmem_quiet();  // Or the correct scope-matched flush when sufficient.

__syncthreads();
nvshmemx_release_smem();
```

Restructure early returns only as needed to guarantee that every CTA thread reaches matching registration and release calls. Never place `nvshmemx_give_smem`, `nvshmemx_release_smem`, or `__syncthreads()` in CTA-divergent control flow.

## Default GMEM Registration and Launch

For GMEM staging, keep the operands and RMA call unchanged; donate an NVSHMEM-owned dynamic-SMEM region. Adapt the application's error handling and launch abstraction rather than replacing them:

```cuda
__global__ void put_kernel(float *remote_dst, const float *local_src,
                           size_t nelems, int peer,
                           size_t nvshmem_smem_bytes) {
    extern __shared__ __align__(16) unsigned char dynamic_smem[];

    nvshmemx_give_smem(dynamic_smem, nvshmem_smem_bytes);
    __syncthreads();

    nvshmemx_float_put_nbi_block(remote_dst, local_src, nelems, peer);
    nvshmem_quiet();  // Keep or adapt the application's required completion.

    __syncthreads();
    nvshmemx_release_smem();
}

size_t nvshmem_smem_bytes =
    (size_t)nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);
size_t total_dynamic_smem = nvshmem_smem_bytes;

cudaFuncSetAttribute(
    put_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    (int)total_dynamic_smem);

put_kernel<<<grid, block, total_dynamic_smem, stream>>>(
    remote_dst, local_src, nelems, peer, nvshmem_smem_bytes);
```

Fit `cudaFuncSetAttribute` into the application's CUDA error-checking convention. Check the device's opt-in shared-memory limit before claiming the launch is valid.

## Coexisting Application Shared Memory

Keep application shared memory outside the NVSHMEM prefix and preserve its alignment. A kernel may retain static application shared memory and add a separate dynamic NVSHMEM region; account for static plus dynamic use when checking the per-block limit and occupancy.

```cuda
static size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

size_t nvshmem_smem_bytes =
    (size_t)nvshmemx_ask_smem(NVSHMEMX_SMEM_RECOMMENDED);
size_t app_offset = align_up(nvshmem_smem_bytes, app_alignment);
size_t total_dynamic_smem = app_offset + app_smem_bytes;

kernel<<<grid, block, total_dynamic_smem, stream>>>(
    args..., nvshmem_smem_bytes, app_offset);

__global__ void kernel(/* application args */,
                       size_t nvshmem_smem_bytes,
                       size_t app_offset) {
    extern __shared__ __align__(16) unsigned char dynamic_smem[];
    void *nvshmem_smem = dynamic_smem;
    float *application_tile =
        reinterpret_cast<float *>(dynamic_smem + app_offset);

    nvshmemx_give_smem(nvshmem_smem, nvshmem_smem_bytes);
    __syncthreads();

    // Existing application work and eligible NVSHMEM operations.
    nvshmem_quiet();  // Or the correct scope-matched flush when sufficient.

    __syncthreads();
    nvshmemx_release_smem();
}
```
