# Compute Sanitizer — Racecheck

> Reference sub-skill of the `compute-sanitizer` skill. Read this after the
> router (`SKILL.md`) routes a **shared-memory race / hazard** symptom here.
> Detects: WAW / WAR / RAW hazards on `__shared__` memory, `cuda::barrier`
> deadlocks, `cp.async` races. Shared memory only — not global races.

## What It Detects

Shared memory (`__shared__`) data access hazards within a thread block:

| Hazard Type | Description |
|---|---|
| WAW (Write-After-Write) | Two threads write the same shared memory byte without synchronization |
| WAR (Write-After-Read) | A write overtakes a read to the same byte (writer ordered before reader) |
| RAW (Read-After-Write) | A read occurs before a write commits to the same byte |
| `cuda::barrier` deadlocks | Threads blocked indefinitely on barrier wait |
| `cp.async` / `memcpy_async` races | Shared memory target accessed before async copy completes |
| Cluster entry/exit races | Remote shared memory access without cluster-wide synchronization (Hopper+) |

**Scope limitation:** racecheck only covers on-chip shared memory. It does NOT detect global memory races. Use iGUARD or SuperCollider for global memory race detection.

## When to Use

- After memcheck passes clean (racecheck does not check memory access errors).
- Any time kernels use `__shared__` memory accessed by multiple threads.
- When debugging non-deterministic results that vary across runs or GPU architectures.
- When porting pre-Volta warp-synchronous code to modern architectures where lock-step execution is not guaranteed.

## Build

racecheck runs on a **vanilla** `nvcc -lineinfo` binary. A
`-fdevice-sanitize=memcheck` binary alters device codegen and degrades
racecheck's hazard analysis — never use one here. General build/setup lives in
the router `SKILL.md`.

## Running the Tool

```bash
compute-sanitizer --tool racecheck [options] ./app [app_args]
```

### Typical invocations

```bash
# Default (analysis report mode)
compute-sanitizer --tool racecheck --report-api-errors=no ./app

# Detailed per-byte hazard reports
compute-sanitizer --tool racecheck --racecheck-report hazard ./app

# Include INFO-level hazards (warp-synchronous programming issues)
compute-sanitizer --tool racecheck --racecheck-detect-level info ./app

# Enable deadlock detection (5-second timeout)
compute-sanitizer --tool racecheck --racecheck-deadlock-timeout 5000 ./app

# Log to file with timeout
timeout 30 compute-sanitizer --tool racecheck --report-api-errors=no \
    --log-file racecheck.log ./app
```

## Key Options

| Option | Values | Default | Purpose |
|---|---|---|---|
| `--racecheck-report` | hazard, analysis, all | analysis | `hazard`: byte-accurate per-thread detail; `analysis`: high-level summary |
| `--racecheck-detect-level` | info, warn, error | warn | Minimum severity to report |
| `--racecheck-deadlock-timeout` | {ms} | 0 (disabled) | Detect `cuda::barrier` deadlocks after N ms. Small values may cause FPs |
| `--racecheck-continue-on-deadlock` | yes, no | yes | Whether to continue past detected deadlocks |
| `--racecheck-memcpy-async` | yes, no | yes | Check `cp.async` / `memcpy_async` races |
| `--racecheck-num-workers` | {N} | 0 (auto) | CPU worker threads for analysis |
| `--racecheck-trace-sync` | yes, no | no | Log synchronization operations (for debugging sync issues) |
| `--num-cuda-barriers` | {N} | 0 (auto) | Override cuda::barrier count if compiler undercounts |
| `--print-level` | info, warn, error, fatal | warn | Minimum print level |
| `--kernel-name` | kns=substring | all | Filter to specific kernels |

## Interpreting Output

### Analysis report (default mode)

```
========= WARNING: Race reported between Write access at RAW()+0xf0 in raceGroupBasic.cu:40
=========     and Read access at RAW()+0x280 in raceGroupBasic.cu:46 [4 hazards]
```

- First word = severity: `INFO`, `WARNING`, or `ERROR`
- Each line: access type (Read/Write), kernel name, instruction offset, file:line
- Bracketed count = number of underlying hazards consolidated into this report
- Severity of the report is the highest severity among its constituent hazards

### Hazard report (`--racecheck-report hazard`)

```
========= ERROR: Potential WAW hazard detected at __shared__ 0x0 in block (0,0,0) :
=========     Write Thread (0,0,0) at WAW()+0x2f0 in raceWAW.cu:20
=========     Write Thread (1,0,0) at WAW()+0x2f0 in raceWAW.cu:20
=========     Current Value : 1, Incoming Value : 2
```

Line-by-line:
1. Severity, hazard type (WAW/WAR/RAW), shared memory byte offset, block index
2. First thread: access type, thread index, kernel+offset, file:line
3. Second thread: same format (ordered chronologically)
4. Data values (for WAW/WAR): `Current Value` = after first access, `Incoming Value` = second thread's write

### Async copy race

Individual hazards bear the mention `(invalid memcpy_async synchronization)` when the target of an async copy was accessed before required commit/wait.

### Cluster entry/exit race (Hopper+)

```
========= Potential invalid __shared__ read of size 4 bytes
=========     at RemoteAccess(int *, int)+0x170 in RaceCluster.cu:10
=========     by thread (0,0,0) in block (0,0,0)
=========     Address 0x1000400 is located in a block that might not have entered yet
```

### Deadlock report

```
========= Deadlock detected in Cuda::barrier wait called by threads starting at (0,0,0) in block (0,0,0), (active mask: 0xffffffff), barrier address: 0x0 at kernel+0x1190 in app.cu:98
```

### Sync trace (`--racecheck-trace-sync yes`)

```
========= Barrier called by threads starting at (32,0,0) in block (1,0,0), (active mask: 0xffffffff) at kernel()+0x2b0 in app.cu:36
```

### Error summary

```
========= ERROR SUMMARY: N errors
```

## Severity Levels

| Level | Meaning | Action |
|---|---|---|
| INFO | No impact on execution, but indicates a potential design issue | Investigate to be safe |
| WARNING | Programming-model hazard, may be intentional (e.g., warp-level tricks) | Fix unless deliberately warp-synchronous |
| ERROR | Very likely data race causing non-deterministic results | Must fix |

On post-Volta architectures, warp-synchronous code without explicit `__syncwarp()` is not safe. Warnings from such patterns must be fixed.

## Log Parsing Quick Reference

```bash
# Any errors?
grep "ERROR SUMMARY" log.txt | grep -v "0 errors"

# Analysis-mode races
grep "Race reported between" log.txt

# Hazard-mode races (by type)
grep "Potential WAW hazard" log.txt
grep "Potential WAR hazard" log.txt
grep "Potential RAW hazard" log.txt

# Async copy races
grep "invalid memcpy_async synchronization" log.txt

# Cluster races
grep "might not have entered yet\|might have already exited" log.txt

# Deadlocks
grep "Deadlock detected" log.txt

# Count hazards per kernel
grep -oP 'at \K\S+\(\)' log.txt | sort | uniq -c | sort -rn
```

## Limitations and Caveats

- **Shared memory only.** Global memory races are invisible to racecheck. Use iGUARD or SuperCollider for those.
- **Do NOT use memcheck-instrumented binaries** (`-fdevice-sanitize=memcheck`) with racecheck. Use a vanilla `-lineinfo` build.
- **`cuda::barrier` tracking** may undercount barriers. If you see `Detected overflow of tracked cuda::barrier structures`, use `--num-cuda-barriers N`.
- **High memory footprint** under heavy concurrent launches. Use `--force-synchronization-limit` if OOM.
- Racecheck **does not terminate kernels** on hazard detection — it only reports. The program continues.
- Very small `--racecheck-deadlock-timeout` values can cause false-positive deadlock reports.
- Env var `NV_COMPUTE_SANITIZER_MAX_RACECHECK_HAZARDS` (default 10M) caps the number of processed hazards — increase if truncation occurs.

## References

- [Compute Sanitizer docs — Racecheck Tool](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#racecheck-tool)
- [Compute Sanitizer samples](https://github.com/NVIDIA/compute-sanitizer-samples)
