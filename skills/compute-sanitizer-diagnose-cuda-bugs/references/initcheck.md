# Compute Sanitizer — Initcheck

> Reference sub-skill of the `compute-sanitizer` skill. Read this after the
> router (`SKILL.md`) routes an **uninitialized / unused memory** symptom here.
> Detects: uninitialized global/shared reads, allocated-but-never-written memory.

## What It Detects

| Error Class | Description |
|---|---|
| Uninitialized global memory reads | Kernel reads device global memory that was never written by a kernel or CUDA memcpy/memset |
| Uninitialized shared memory reads | Kernel reads `__shared__` memory that was never written (opt-in) |
| Unused memory allocations | Allocated memory (global or shared) that was never written to |

## When to Use

- After memcheck passes clean (initcheck does not check memory access errors).
- When GPU computation produces incorrect/garbage results and you suspect uninitialized memory.
- When auditing memory utilization — find allocations that are allocated but never used.
- Complement to host-side tools: initcheck catches device-side uninit reads that Valgrind/ASan cannot see.

## Build

initcheck requires an uninstrumented `nvcc -lineinfo` binary; a
`-fdevice-sanitize=memcheck` build can mask the uninitialized reads it hunts
for. See the router `SKILL.md` for general build and setup.

## Running the Tool

```bash
compute-sanitizer --tool initcheck [options] ./app [app_args]
```

### Typical invocations

```bash
# Check global memory (default)
compute-sanitizer --tool initcheck --report-api-errors=no ./app

# Check both global and shared memory
compute-sanitizer --tool initcheck --initcheck-address-space all ./app

# Shared memory only
compute-sanitizer --tool initcheck --initcheck-address-space shared ./app

# Also detect unused memory
compute-sanitizer --tool initcheck --track-unused-memory ./app

# Unused memory with threshold (suppress reports under 50% unused)
compute-sanitizer --tool initcheck --track-unused-memory \
    --unused-memory-threshold 50 ./app

# Log to file with timeout
timeout 30 compute-sanitizer --tool initcheck --report-api-errors=no \
    --log-file initcheck.log ./app
```

## Key Options

| Option | Values | Default | Purpose |
|---|---|---|---|
| `--initcheck-address-space` | global, shared, all | global | Which address space to check |
| `--track-unused-memory` | N/A | disabled | Report memory that was allocated but never written |
| `--unused-memory-threshold` | 0-100 | 0 | Minimum unused percentage to report (suppresses low-percentage reports) |
| `--unused-shared-memory-report-limit` | {N} | 32 | Max unused shared memory reports per launch |
| `--check-api-memory-access` | yes, no | yes | Check `cudaMemcpy`/`cudaMemset` operations |
| `--check-optix` | N/A | disabled | Track OptiX kernel launches |
| `--report-api-errors` | all, extended, explicit, no | extended | CUDA API error reporting level |
| `--kernel-name` | kns=substring | all | Filter to specific kernels |
| `--log-file` | {path} | stdout | Redirect output to file |

## Interpreting Output

### Uninitialized global memory read

```
========= Uninitialized __global__ memory read of size 4 bytes
=========     at kernel_name()+0x120 in app.cu:42
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x7fed9f400000
```

Line-by-line:
1. Address space (`__global__` or `__shared__`), access size
2. Kernel name, instruction offset, file:line (if `-lineinfo`)
3. Thread and block indices
4. Memory address that was read without initialization

### Uninitialized shared memory read

```
========= Uninitialized __shared__ memory read of size 4 bytes
=========     at KernelSharedError()+0x1a0 in uninitialized_shared.cu:7
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x0
```

Requires `--initcheck-address-space shared` or `--initcheck-address-space all`.

### Unused memory report

```
=========  Unused __shared__ memory in allocation of size 100 bytes in block (0,0,0)
=========     Not written 80 bytes at offset 0x14 (0x14)
=========     These unused bytes represent 80% of the shared memory.

=========  Unused __global__ memory in allocation 0x7fed9f400000 of size 100 bytes
=========     Not written 80 bytes at offset 0x14 (0x7fed9f400014)
=========     80% of allocation were unused.
```

Reports include allocation address/size, byte ranges not written, and percentage unused. Note: compiler-inserted padding may inflate reported allocation sizes.

### Error summary

```
========= ERROR SUMMARY: N errors
```

## Log Parsing Quick Reference

```bash
# Any errors?
grep "ERROR SUMMARY" log.txt | grep -v "0 errors"

# Uninitialized reads (both global and shared)
grep "Uninitialized" log.txt

# Uninitialized global reads only
grep "Uninitialized __global__" log.txt

# Uninitialized shared reads only
grep "Uninitialized __shared__" log.txt

# Unused memory reports
grep "Unused" log.txt

# Extract kernel + file:line from errors
grep -E "^\s*=========\s+at " log.txt

# Summary of unused percentages
grep -oP '\d+% of' log.txt
```

## Limitations and Caveats

- **Does not check memory access errors.** Run memcheck first to ensure the application is error-free.
- **Do NOT use memcheck-instrumented binaries** with initcheck. Use a vanilla `-lineinfo` build.
- **Shared memory padding**: the compiler may insert extra padding in shared memory allocations, leading to unused-memory reports for bytes the programmer never intended to use.
- **OptiX**: requires `--check-optix` flag; shared memory address space not supported with OptiX.
- **Memory footprint** can be large under heavy concurrent kernel launches. Use `--force-synchronization-limit` if OOM.
- Initcheck tracks writes from device-side kernels and CUDA memcpy/memset API calls. Memory initialized by other means (e.g., DMA, peer access) may produce false positives.

## References

- [Compute Sanitizer docs — Initcheck Tool](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#initcheck-tool)
- [Compute Sanitizer samples](https://github.com/NVIDIA/compute-sanitizer-samples)
