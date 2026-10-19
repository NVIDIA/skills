# Direct Application-SMEM Put Path

Apply this path only after completing the shared registration and launch requirements selected by the main skill.

Apply this path only after explicit user opt-in. It is available only when an application-owned shared-memory tile already supplies a put source and the intermediate SMEM-to-GMEM copy and GMEM buffer are proven to have no other producer, consumer, alias, synchronization, completion, or lifetime role.

If the selected source cannot prove that the GMEM buffer is not externally observable—for example, it is a kernel or host parameter, escapes the translation unit, or may have uninspectable consumers—ask the user to confirm that it has no external role before removing it. Keep the managed GMEM-staging path until confirmed.

- Keep the application tile outside the NVSHMEM-registered prefix and pass the application tile pointer, not the prefix pointer, to the NVSHMEM put.
- Remove only the proven-redundant copy and storage; preserve all other application movement and completion behavior.
- For NVSHMEM 3.7.x, apply this path only to put sources. Documented TMA-backed gets use NVSHMEM's internal per-CTA staging. A direct shared-memory get destination requires exact target-version NVSHMEM documentation.
- Use `NVSHMEMX_SMEM_RECOMMENDED` if any selected operation needs GMEM staging. Query `NVSHMEMX_SMEM_BARRIERS_ONLY` only when every selected operation is a direct application-SMEM put source.

After the tile producers synchronize, issue the SM90+ async-proxy fence immediately before the put:

```cuda
for (size_t i = threadIdx.x; i < tile_elems; i += blockDim.x) {
    application_tile[i] = produce_value(i);
}
__syncthreads();

#if __CUDA_ARCH__ >= 900
asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif

nvshmemx_float_put_nbi_block(
    remote_dst, application_tile, tile_elems, peer);
```

Do not use this fence for ordinary GMEM staging. It orders writes to an application shared-memory source for the TMA async proxy.
