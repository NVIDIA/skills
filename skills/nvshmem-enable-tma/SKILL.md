---
name: nvshmem-enable-tma
description: Prepare or review NVSHMEM CUDA kernels for TMA SMEM registration and direct-SMEM transfers. Do not use for unrelated CUDA tuning.
license: Apache-2.0
metadata:
  version: "1.0.0"
  author: NVIDIA NVSHMEM Team <nvshmem@nvidia.com>
  tags:
    - nvshmem
    - tma
---

# Enable TMA in NVSHMEM Applications

## Purpose

Prepare an application for NVSHMEM's TMA-backed device-side transfers without changing its put/get APIs or unrelated kernels. Assess the target, choose the correct shared-memory path, make the smallest safe change, and explain launch-time requirements.

## Prerequisites

- Require readable CUDA/C++ source for a patch or code review. A static assessment needs no GPU, installed NVSHMEM runtime, or cluster access.
- Require an NVSHMEM 3.7.0-or-newer development environment and CUDA toolchain only when compiling.
- Require SM90-or-newer GPUs and peer GPU load/store reachability such as NVLink only when validating TMA execution or performance.
- CFT requires TMA.

## Reference Routing

Before changing code, read [registration-and-launch.md](references/registration-and-launch.md) completely and then every reference selected below. Use the references as shapes, not blind replacements: preserve the application's error handling, launch abstraction, completion contract, and shared-memory layout.

| Situation | Required additional reference |
| --- | --- |
| Global-memory operands, or staging copy/buffer is not proven redundant | [gmem-staging.md](references/gmem-staging.md) |
| Explicitly opted-in shared-memory put source with a proven-redundant GMEM copy | [direct-smem-put.md](references/direct-smem-put.md) |
| Compile, run, diagnose, or report validation | [validation-and-troubleshooting.md](references/validation-and-troubleshooting.md) |

For a read-only assessment or review, read only the references needed to evaluate the proposed path. Do not make a code change until the complete registration reference and every selected path reference have been read.

## Inputs

- **Required:** Application source or source path, and the requested outcome: assessment, patch, review, or debugging.
- **Optional:** Derive NVSHMEM version, CUDA architecture, topology, kernel launch sites, and launch configuration from the project or environment when possible.
- **Conditional for execution:** Obtain build command, launch command, and target hardware access only when the user asks to compile or run validation.
- Ask only for facts that cannot be inspected and materially affect the result. Do not require topology details merely to prepare code; mark performance benefit unverified when topology is unknown.

## Scope

- Identify exactly one user-opted-in CUDA translation unit and one NVSHMEM communication kernel before editing. Its required launch code is the complete change boundary.
- Derive the target from the user's named file/kernel or attachment metadata, not from files discovered by globbing. A singular request does not authorize editing every eligible kernel.
- Preserve all other kernels, host-only NVSHMEM use, and unrelated launch code. Never modify adjacent candidates merely because they contain NVSHMEM calls.
- Expand beyond one kernel only when the user explicitly names multiple targets. Apply this scope check to each target independently.

Before editing, answer every question below. Stop without editing if a required answer is no or unverified.

- **Explicit scope:** Is this the selected file and kernel?
- **Eligible operation:** Does the selected kernel's device call graph reach a TMA-relevant NVSHMEM operation?
- **Necessary edit:** Is every planned change needed for registration, lifetime, direct-SMEM dataflow, or launch sizing?
- **Proven redundancy:** If removing a staging copy or buffer, is it proven to have no producer, consumer, alias, synchronization, completion, or lifetime role? If the answer is unverified because the GMEM buffer is a parameter, escapes the selected translation unit, or has uninspectable consumers, ask the user whether it is externally observable before editing. Do not infer redundancy.
- **No speculation:** Does the selected target require application-side TMA registration for the operation being prepared?
- **Surgical launch change:** Can the existing grid, block, stream, arguments, and error handling remain intact apart from required registration parameters and dynamic-SMEM bytes?

After editing, inspect the changed-file list and diff. Remove only your own out-of-scope edits before reporting completion; the changed application file set must contain only the opted-in target.

## Instructions

### 1. Establish the Boundary

- Treat an assessment or review as read-only.
- Treat a request to enable, prepare, update, or fix TMA as permission to edit only the selected kernel and in-scope launch configuration.
- Inspect the selected kernel's device call graph for device-side point-to-point put, get, put-with-signal, and their typed/thread/warp/block variants.

### 2. Decide Whether TMA Is Useful and Available

State these conclusions before or alongside a patch:

- Require NVSHMEM 3.7.0 or newer. Stop a requested conversion for an older confirmed version and recommend upgrading.
- Require SM90 or newer for TMA execution. With `NVSHMEM_TMA_POLICY=ENABLE`, older GPUs preserve correctness through the regular path but receive no TMA benefit.
- TMA helps only peer-reachable GPU memory paths through the GPU load/store fabric, normally NVLink. Do not recommend it for IB, RoCE, EFA, or another network transport.
- It is promising for large point-to-point transfers, low communication-thread counts, fused kernels, or tiles already in shared memory. Never promise a speedup without measurement.

When version, architecture, or topology is unknown, prepare code only if requested and label the corresponding conclusion unverified.

### 3. Classify Each Operation

Inspect local operand spaces, application shared memory, staging-copy dataflow, dynamic/static shared memory, alignment, early returns, CTA-uniform control flow, existing fences/flushes/quiet calls, and possible concurrent staged callers in a CTA.

| Situation | Path |
| --- | --- |
| Global-memory application operands, or no proven-redundant staging | NVSHMEM-managed GMEM staging |
| Shared-memory put source with a proven-redundant SMEM-to-GMEM copy | Offer direct shared-memory put as an explicit opt-in |
| Shared-memory get destination | Keep the documented staging path |

Default to NVSHMEM-managed staging. Never introduce a new application shared-memory layout solely to force a direct-operand path.

### 4. Apply the Selected References

Follow the complete registration/lifetime protocol in the shared registration reference. Then apply the selected path reference. Preserve existing completion semantics and do not hard-code the current recommended shared-memory byte count.

### 5. Validate and Report

Read the validation reference for all compile, run, diagnosis, and validation-report work. Lead the final report with one of:

- `TMA is useful and the application is prepared`
- `The application is prepared, but TMA benefit is unverified`
- `TMA is not useful or not supported for this target`

Use this report template:

```text
Outcome: <one required outcome line above>
Changed scope: <the selected file, kernel, and launch site; confirm no adjacent files changed>
Selected path: <NVSHMEM-managed GMEM staging | explicitly opted-in direct-SMEM put>
Policy precondition: Set NVSHMEM_TMA_POLICY=ENABLE before nvshmem_init*; <where to set it>
Eligibility: NVSHMEM <version, must be >= 3.7.0>; <SM90+ evidence>; <NVLink/peer load-store evidence>
Operation constraints: <alignment, byte count, scope/concurrency, completion, shared-memory sizing>
Fallback: ENABLE preserves the regular path for ineligible operations; registration alone does not guarantee TMA routing
Unverified: <topology, occupancy, or none>
Validation: <checks run and exact remaining test commands>
```

Mention any direct-SMEM candidate deliberately not applied because it was outside scope or its redundancy was not proven.

## Limitations

- Do not convert host-side RMA, collectives without an applicable TMA path, network-transport operations, or remote shared-memory targets.
- Do not claim registration guarantees TMA routing; policy, architecture, topology, operand space, alignment, size, scope, and runtime limits determine each transfer.

## Troubleshooting

For configuration failures, incorrect results, hangs near shared-memory release, or an unselected TMA path, read [validation-and-troubleshooting.md](references/validation-and-troubleshooting.md). Treat regular-path fallback under `NVSHMEM_TMA_POLICY=ENABLE` as valid behavior; do not diagnose it as a correctness failure.

## Examples

```text
Use $nvshmem-enable-tma to prepare app.cu for NVSHMEM TMA on H100 NVLink while leaving non-NVSHMEM kernels unchanged.

Use $nvshmem-enable-tma to inspect a shared-tile-to-GMEM-to-put sequence and, if the GMEM copy is provably redundant, pass the existing shared tile directly.

Use $nvshmem-enable-tma to assess an NVSHMEM 3.6 application running on A100 peers connected only through InfiniBand.
```

<!-- ci trigger -->
