# Compute Sanitizer — Memcheck

> Reference sub-skill of the `compute-sanitizer` skill. Read this after the
> router (`SKILL.md`) routes a **memory-error** symptom here.
> Detects: OOB / misaligned access, malloc/free errors, CUDA API errors, leaks.

## What It Detects

| Error Class | Location | Precision |
|---|---|---|
| OOB / misaligned access (global, local, shared, atomic) | Device | Precise |
| Hardware exceptions | Device | Imprecise |
| Device-side `malloc()`/`free()` errors (double free, invalid pointer) | Device | Precise |
| CUDA API call failures | Host | Precise |
| `cudaMalloc` memory leaks | Host | Precise |
| Device heap (`malloc()` in kernel) leaks | Device | Imprecise |
| Stream-ordered allocation races (use-before-alloc, use-after-free) | Host | Precise |
| Tensor Memory (TCMMA) guardrail violations (Blackwell+) | Device | Precise |

## When to Use

- First tool to run in any sanitizer battery — other tools (racecheck, initcheck, synccheck) assume the program is free of memory errors.
- Use **binary-instrumentation mode** (default) for quick checks without recompilation.
- Use **compile-time instrumentation mode** when adjacent-allocation overflows need detection or when lower runtime overhead matters.

## Prerequisites / Build

### Minimal (binary instrumentation — no recompilation)

Add `-lineinfo` for source attribution in error reports:

```bash
nvcc -lineinfo -o app app.cu
```

For host-side backtraces with function names on Linux, add `-Xcompiler -rdynamic`.

### Compile-Time Instrumentation (enhanced detection)

```bash
nvcc -fdevice-sanitize=memcheck -lineinfo -o app app.cu
```

This converts pointers to fat pointers carrying base+bounds metadata. Catches overflows into contiguous allocations that binary-only mode misses (e.g., `array` overrunning into an adjacent `buffer` allocation).

**Caveats of compile-time mode:**
- Extra register and stack usage — kernels may hit "too many resources requested for launch." Mitigate with `__launch_bounds__` decorations or `--maxregcount`.
- Does not support HMM memory allocations (causes false positives).
- Undefined behavior (including race conditions) can cause unexpected instrumentation results.
- Compiler and compute-sanitizer must come from the same CUDA toolkit.
- Avoid combining with `-G` (debug) — use `-lineinfo` only.

**Hybrid instrumentation** is supported: different `.cu` files in the same binary can mix compile-time and binary-only instrumentation. The tool auto-detects which mode applies per compilation unit.

## Running the Tool

Memcheck is the default tool — `--tool memcheck` is optional:

```bash
compute-sanitizer [--tool memcheck] [options] ./app [app_args]
```

### Typical invocations

```bash
# Basic run (binary instrumentation)
compute-sanitizer --report-api-errors=no ./app

# With compile-time instrumented binary
compute-sanitizer --tool memcheck ./app_sanitized

# Full battery: leak check + stream-ordered race detection
compute-sanitizer --leak-check full --track-stream-ordered-races all ./app

# Log to file
compute-sanitizer --log-file memcheck.log ./app

# Limit output noise
compute-sanitizer --print-limit 50 --report-api-errors no ./app

# With timeout wrapper
timeout 30 compute-sanitizer --tool memcheck --report-api-errors=no ./app 2>&1
```

## Key Options

| Option | Values | Default | Purpose |
|---|---|---|---|
| `--tool` | memcheck | memcheck | Select tool (already default) |
| `--report-api-errors` | all, extended, explicit, no | extended | CUDA API error reporting level |
| `--leak-check` | full, no | no | Report unfreed `cudaMalloc` allocations |
| `--padding` | {N bytes} | 0 | Add invalid padding between allocations — improves OOB detection without recompilation |
| `--track-stream-ordered-races` | all, use-before-alloc, use-after-free, no | all | Detect stream-ordered allocation races |
| `--check-device-heap` | yes, no | yes | Check device-side `malloc()`/`free()` |
| `--destroy-on-device-error` | context, kernel | context | `kernel` allows remaining work to proceed after an error |
| `--log-file` | {path} | stdout | Redirect output to file |
| `--print-limit` | {N} | 100 | Max errors to print (0 = unlimited) |
| `--error-exitcode` | {N} | 0 | Exit code when errors found (for CI integration) |
| `--kernel-name` | kns=substring | all | Filter to specific kernels |
| `--show-backtrace` | yes, host, device, no | yes | Control backtrace display |
| `--generate-coredump` | N/A | disabled | Generate CUDA coredump on first error (loadable in `cuda-gdb`) |
| `--xml` | N/A | disabled | Emit XML output (use with `--save`) |

## Interpreting Output

### Memory access error (precise)

```
========= Invalid __global__ write of size 4 bytes
=========     at unaligned_kernel():0x160 in memcheck_demo.cu:6
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x7f6510c00001 is misaligned
```

Line-by-line:
1. Memory segment (`__global__`/`__shared__`/`__local__`), access type (read/write), size
2. Kernel name, instruction offset, source file:line (if `-lineinfo` used)
3. Thread and block indices
4. Address and error type: `is misaligned` or `is out of bounds` (+ distance to nearest allocation)

### Compile-time instrumentation error (adjacent allocation caught)

```
========= Invalid __global__ read of size 4 bytes
=========     at scaleArray(float *, unsigned long, float)+0x150
=========     by thread (512,0,0) in block (0,0,0)
=========     Access to 0x7b3566000800 is out of bounds
=========     and is inside the nearest allocation at 0x7b3566000800 of size 2,048 bytes
```

The phrase "is inside the nearest allocation" indicates compile-time base+bounds caught an overflow that binary-only mode would miss.

### Malloc/free error

```
========= Malloc/Free error encountered : Double free
=========     at 0x79d8
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x400aff920
```

Types: `Double free`, `Invalid pointer to free`, `Heap corruption`.

### Leak report

```
========= Leaked 64 bytes at 0x400200200
```

Requires `--leak-check full`. Device heap leaks show `on the device heap`.

### CUDA API error

```
========= Program hit invalid copy direction for memcpy (error 21) on CUDA API call to cudaMemcpy.
```

### Error summary

Every run ends with:
```
========= ERROR SUMMARY: N errors
```

## Log Parsing Quick Reference

```bash
# Did any errors occur?
grep "ERROR SUMMARY" log.txt | grep -v "0 errors"

# Memory access errors
grep -E "Invalid __(global|shared|local)__" log.txt

# Malloc/free errors
grep "Malloc/Free error" log.txt

# Leaks
grep "Leaked" log.txt

# CUDA API errors
grep "Program hit" log.txt

# Hardware exceptions
grep "Hardware exception" log.txt

# Stream-ordered races
grep -E "use-before-alloc|use-after-free" log.txt

# Extract kernel + file:line from errors
grep -E "^\s*=========\s+at " log.txt
```

## Coredump → cuda-gdb (post-mortem of the faulting kernel)

When a fault is hard to reproduce or you want to inspect GPU state at the moment
of the error, generate a CUDA coredump and open it in a debugger:

```bash
# Dump on the first error, then abort. Optionally name the file.
compute-sanitizer --tool memcheck --generate-coredump yes \
    [--coredump-name core.nvcudmp] ./app
```

```text
(cuda-gdb) target cudacore core.nvcudmp     # Linux: load the dump
```

- `--coredump-name` sets the filename; `--coredump-behavior full` (default) also
  writes a CPU coredump and aborts, `exit` just exits.
- Set `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1` in the environment for the same effect
  without the flag.
- **Restrictions:** only threads that hit the error are inspectable, and the
  **racecheck tool does not support coredumps** (works with memcheck / initcheck /
  synccheck). On Windows, open the dump in Nsight Visual Studio Edition.

## Making imprecise errors precise

Hardware-reported (imprecise) errors may not attribute to a `file:line`. Re-run
with `--force-blocking-launches` to serialize kernel launches — this usually turns
an imprecise error into a precise one (at the cost of losing concurrency). It also
lowers memory pressure, as does `--force-synchronization-limit N`.

## Limitations and Caveats

- **Binary-only mode misses adjacent-allocation overflows.** Use `--padding N` or compile-time instrumentation (`-fdevice-sanitize=memcheck`) to catch them.
- **Compile-time mode does not support HMM** — false positives on HMM allocations.
- **High memory footprint** under heavy concurrent kernel launches. Use `--force-synchronization-limit N` or `--force-blocking-launches` to reduce pressure.
- **`-G` (full debug) significantly increases register/stack pressure** on top of instrumentation. Prefer `-lineinfo`.
- Memcheck **terminates the CUDA context** on memory access errors by default. Use `--destroy-on-device-error kernel` to only terminate the offending kernel and let the rest continue.
- Racecheck, initcheck, and synccheck assume memcheck-clean code. Always run memcheck first.

## References

- [Compute Sanitizer docs](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
- [Compile-Time Instrumentation blog post](https://developer.nvidia.com/blog/better-bug-detection-how-compile-time-instrumentation-for-compute-sanitizer-enhances-memory-safety/)
- [Compute Sanitizer samples](https://github.com/NVIDIA/compute-sanitizer-samples)
