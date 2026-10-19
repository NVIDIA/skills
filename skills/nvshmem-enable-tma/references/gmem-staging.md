# NVSHMEM-Managed GMEM Staging Path

Apply this path only after completing the shared registration and launch requirements selected by the main skill.

Use this default path for global-memory application operands or when an application staging copy is not proven redundant.

- Keep existing NVSHMEM function names, operands, and symmetric remote pointers unchanged.
- Let NVSHMEM stage eligible global-memory transfers through the donated shared-memory prefix.
- Preserve existing ordering and completion semantics. Add a final completion operation before release only when pending work can still use the registered region.
- For NVSHMEM 3.7.x, serialize staged TMA calls within a CTA when independent thread- or warp-scoped callers could overlap. This applies to all TMA-backed gets and puts with global-memory sources.

For NVSHMEM 3.7.x, TMA routing requires 16-byte-aligned source and destination operands and a byte count divisible by 16. Block-scoped staged puts also require at least two warps. Under `NVSHMEM_TMA_POLICY=ENABLE`, ineligible operations retain regular NVSHMEM behavior rather than becoming correctness failures.

Do not add `fence.proxy.async.shared::cta` to this path; the fence is for an application-owned shared-memory put source.
