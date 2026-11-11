# TMA Validation and Troubleshooting

Read this reference when compiling, running, diagnosing, or reporting validation for a prepared TMA application.

## Configuration and Validation

- Set `NVSHMEM_TMA_POLICY=ENABLE` before any `nvshmem_init*` call. Prefer a launcher or process-environment setting. `FORCE` is diagnostic only; it does not make every transfer TMA-eligible.
- Confirm remote operands are peer global-memory objects, normally NVSHMEM symmetric allocations. Never target remote shared memory.
- Statically confirm that every thread in each participating CTA reaches both `nvshmemx_give_smem` and `nvshmemx_release_smem` on a CTA-uniform path, with identical `give_smem` pointer and size arguments across the CTA. Reject leader-guarded calls even when following the incorrect NVSHMEM 3.7.0, 3.7.1, or 3.7.2 documentation.
- Build or statically validate every changed translation unit when the environment permits.
- Run existing correctness tests with `NVSHMEM_TMA_POLICY=ENABLE`, then compare with `DISABLE` when execution is available.
- Test aligned and unaligned sizes, completion behavior, multi-CTA launches, and pre-existing application shared-memory paths relevant to the change.
- Report unrun tests and unavailable topology or hardware checks explicitly.

## Troubleshooting

| Symptom | Likely cause | Response |
| --- | --- | --- |
| TMA is never selected | Policy disabled, NVSHMEM older than 3.7.0, pre-SM90 GPU, non-NVLink peer, missing CTA registration, or ineligible alignment/size/scope | Check each eligibility gate, use `ENABLE`, and compare with `DISABLE`; regular-path fallback remains valid. |
| Kernel launch fails after adding dynamic SMEM | Static plus dynamic SMEM exceeds the limit, or the opt-in attribute was not set | Query the device limit, check `cudaFuncSetAttribute`, preserve alignment, and reconsider the SMEM amount only with the documented tradeoff. |
| Kernel hangs or data is wrong near registration or release | Only a subset of CTA threads called `give_smem` or `release_smem`, CTA divergence, skipped release, or release before TMA work drains | On a CTA-uniform path, enforce `all threads: give_smem -> __syncthreads -> operations -> flush/quiet -> __syncthreads -> all threads: release_smem`; remove leader guards and bypassing early returns. |
| Direct-SMEM conversion corrupts data | The removed GMEM path had another role, the tile overlaps the prefix, or the fence is absent | Restore nonredundant dataflow, separate the regions, and fence tile producers before a direct put. |
