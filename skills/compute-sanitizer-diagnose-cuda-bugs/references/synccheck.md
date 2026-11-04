# Compute Sanitizer — Synccheck

> Reference sub-skill of the `compute-sanitizer` skill. Read this after the
> router (`SKILL.md`) routes a **synchronization / barrier** symptom here.
> Detects: divergent `__syncthreads()` / `__syncwarp()`, invalid barrier args,
> warpgroup MMA (wgmma) sequence errors, `cuda::barrier` misuse.

## What It Detects

| Error Class | Description |
|---|---|
| Divergent threads in block | `__syncthreads()` / block barrier reached by divergent threads |
| Divergent threads in warp | `__syncwarp()` / warp barrier reached with divergent threads |
| Invalid arguments | Barrier primitive called with incorrect arguments (e.g., `__syncwarp()` mask doesn't match active threads) |
| Warpgroup MMA sequence errors | `wgmma` instructions executed with inconsistent predicates, ordering, or thread activity across a warpgroup (sm_90a) |
| `cuda::barrier` violations | Incorrect barrier usage detected through barrier tracking |

## When to Use

- After memcheck passes clean (synccheck does not check memory access errors).
- When kernels hang or deadlock — divergent `__syncthreads()` is a common cause.
- When porting pre-Volta warp-synchronous code to modern architectures.
- When using `wgmma` instructions on Hopper (sm_90a) to verify warpgroup discipline.
- When using `cuda::barrier` on Ampere+ to verify correct barrier usage.

## Build

synccheck's barrier and warpgroup (wgmma) tracking is only valid on an
uninstrumented `nvcc -lineinfo` binary, not a `-fdevice-sanitize=memcheck`
build. Build flags and setup are covered in the router `SKILL.md`.

## Running the Tool

```bash
compute-sanitizer --tool synccheck [options] ./app [app_args]
```

### Typical invocations

```bash
# Basic synccheck
compute-sanitizer --tool synccheck --report-api-errors=no ./app

# With wgmma checks (Hopper sm_90a)
compute-sanitizer --tool synccheck --check-warpgroup-mma yes ./app

# Specify cuda::barrier count if auto-detection undercounts
compute-sanitizer --tool synccheck --num-cuda-barriers 4 ./app

# Log to file with timeout
timeout 30 compute-sanitizer --tool synccheck --report-api-errors=no \
    --log-file synccheck.log ./app
```

## Key Options

| Option | Values | Default | Purpose |
|---|---|---|---|
| `--check-warpgroup-mma` | yes, no | yes | Enable wgmma sequence checks (sm_90a) |
| `--num-cuda-barriers` | {N} | 0 (auto) | Override cuda::barrier count per block |
| `--missing-barrier-init-is-fatal` | yes, no | yes | Whether uninitialized `cuda::barrier` terminates the warp |
| `--report-api-errors` | all, extended, explicit, no | extended | CUDA API error reporting level |
| `--kernel-name` | kns=substring | all | Filter to specific kernels |
| `--destroy-on-device-error` | context, kernel | context | `kernel` terminates only the offending kernel |
| `--log-file` | {path} | stdout | Redirect output to file |

## Interpreting Output

### Divergent threads in warp

```
========= Barrier error detected. Divergent thread(s) in warp
=========     at ThreadDivergence(int *, int)+0xf0 in divergence.cu:79
=========     by thread (37,0,0) in block (0,0,0)
```

This means `__syncwarp()` (or equivalent) was executed inside a conditional where not all warp threads reach the barrier. Thread 37 diverged.

### Divergent threads in block

```
========= Barrier error detected. Divergent thread(s) in block
=========     at BlockKernel()+0x200 in sync_demo.cu:15
=========     by thread (128,0,0) in block (0,0,0)
```

`__syncthreads()` inside a conditional that doesn't evaluate identically across all threads in the block.

### Invalid arguments

```
========= Barrier error detected. Invalid arguments
=========     at WarpKernel()+0x180 in sync_demo.cu:25
=========     by thread (5,0,0) in block (0,0,0)
```

Typically: `__syncwarp(mask)` where the mask doesn't declare all threads that actually reach it.

### Warpgroup MMA sequence error (Hopper)

```
========= Warpgroup MMA sequence error detected. <specific description>
=========     at MMAKernel()+0x300 in mma_demo.cu:50
=========     by thread (0,0,0) in block (0,0,0)
```

Each warp in a warpgroup must execute the same `wgmma` instructions in the same order with identical predicates. This fires when that contract is violated.

### cuda::barrier overflow warning

```
========= Warning: Detected overflow of tracked cuda::barrier structures. Results might be incorrect. Try using --num-cuda-barriers to fix the issue
```

Fix by specifying `--num-cuda-barriers N` with the actual barrier count.

### Error summary

```
========= ERROR SUMMARY: N errors
```

## Error Actions

Synccheck errors **terminate the CUDA context** by default (same as memcheck). Use `--destroy-on-device-error kernel` to only terminate the offending kernel and allow remaining work to continue.

## Log Parsing Quick Reference

```bash
# Any errors?
grep "ERROR SUMMARY" log.txt | grep -v "0 errors"

# All barrier errors
grep "Barrier error detected" log.txt

# Divergent thread errors specifically
grep "Divergent thread" log.txt

# Block-level vs warp-level divergence
grep "Divergent thread(s) in block" log.txt
grep "Divergent thread(s) in warp" log.txt

# Invalid arguments
grep "Invalid arguments" log.txt

# Warpgroup MMA errors
grep "Warpgroup MMA sequence error" log.txt

# cuda::barrier warnings
grep "overflow of tracked cuda::barrier" log.txt

# Extract kernel + file:line from errors
grep -E "^\s*=========\s+at " log.txt
```

## Limitations and Caveats

- **Does not check memory access errors.** Run memcheck first.
- **Do NOT use memcheck-instrumented binaries** with synccheck. Use a vanilla `-lineinfo` build.
- **`__syncwarp()` mask coverage gap**: synccheck detects when threads reaching `__syncwarp()` don't declare themselves in the mask, but it does NOT detect when threads declared in the mask fail to reach the `__syncwarp()`.
- **`cuda::barrier` tracking** may undercount. Use `--num-cuda-barriers N` if warned.
- **`wgmma` checks** only apply to sm_90a (Hopper). Requires `--check-warpgroup-mma yes` (on by default).
- Synccheck terminates the CUDA context on error by default — downstream kernels will fail.

## References

- [Compute Sanitizer docs — Synccheck Tool](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#synccheck-tool)
- [Compute Sanitizer samples](https://github.com/NVIDIA/compute-sanitizer-samples)
