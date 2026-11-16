# Profiling Guide (Nsys Orchestrator)

All profile artifacts live under `{source_dir}/.nsight-ai-port/profiles/`.
Use **nsys only** — no perf or other profilers.

Copy `.nsight-ai-port/.gitignore` from
[assets/nsight-port-gitignore-template](../../../assets/nsight-port-gitignore-template)
(or the template contents) so binary `.nsys-rep` / `.sqlite` files are
not committed — track text distillations only.

## Directory Layout

```text
.nsight-ai-port/profiles/
├── round_0/                   ← single mode
│   ├── baseline.nsys-rep      ← local only (gitignored)
│   ├── nvtx_sum.csv
│   ├── extraction_meta.txt
│   ├── gaps.txt
│   └── summary.txt
└── round_N/
    └── ...
```

Record the active round in `progress.json:profile_rounds[]`.

**Sequence mode** — namespace by problem id so problems never share a
directory: `.nsight-ai-port/profiles/{problem_id}/round_N/`. The path
examples below show `profiles/round_N/`; in sequence mode prepend the
active `{problem_id}/`.

## Problem Run Command

Nsys profiles the **active problem's** performance case only — its
`run_performance`, built with the matching `build_performance`.
Resolve both paths by matching `progress.json:current_problem` to
`config.json:problems[]`. Never profile the correctness scripts
(`build_correctness.sh` / `run_correctness.sh`); those run the full
numerical test suite, not a single representative workload.

The examples below use `P_BUILD_PERFORMANCE`,
`P_RUN_PERFORMANCE`, and `PROFILE_ROOT` for the resolved paths.
`PROFILE_ROOT` is `.nsight-ai-port/profiles` in single mode and
`.nsight-ai-port/profiles/{problem_id}` in sequence mode.

## CPU Sampling Prerequisites (Round 0)

Round 0 uses `nsys profile --sample=process-tree` to find hotspot functions
**before** NVTX decoration. CPU sampling is **required** — there is
**no fallback** to static codebase planning or reading all sources.

### Preflight

During setup, run
`scripts/profile-check.sh {to_lang}-{framework}`. It runs
`nsys status --environment` and checks
`/proc/sys/kernel/perf_event_paranoid`, `python3` + `sqlite3`, and
the flavor compiler.

If CPU sampling is unavailable **STOP** and report to the user:

1. Show output of `nsys status --environment`.
2. Explain likely causes (from the
   [Nsight Systems Linux requirements](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html#requirements-for-x86-64-and-arm-sbsa-targets-on-linux)):
   - **Paranoid level** — `/proc/sys/kernel/perf_event_paranoid` must
     be ≤ 2 for Linux Perf CPU IP/backtrace sampling (`system-wide`
     needs ≤ 0 or root). `nsys status --environment` reports it as
     "Linux Kernel Paranoid Level". With the user's approval, lower it
     until reboot with
     `sudo sh -c 'echo 2 > /proc/sys/kernel/perf_event_paranoid'`, or
     persist it with `kernel.perf_event_paranoid=2` in
     `/etc/sysctl.d/`.
   - **Docker/containers** — default seccomp may block
     `perf_event_open`; ask the user or platform administrator for the narrow
     seccomp/capability adjustment required by the profiler. Do not broaden all
     container permissions.
   - **Kernel** — need ≥ 4.3 (RedHat backport 3.10.0-693+).
   - **VMs** — some block hardware sampling triggers (software
     fallback may still work).
3. Suggest fixes; let the user iterate (lower paranoid level, run
   outside container, bare metal, etc.) or ask the agent to help
   adjust the environment.
4. **Do not proceed** with Round 0 nsys profiling or NVTX placement
   until a baseline `--sample=process-tree` run produces usable hotspot
   data.

After the user fixes the environment, re-run
`scripts/profile-check.sh {to_lang}-{framework}` and retry.

## Round 0: Baseline CPU Sample (No NVTX)

```bash
cd "$NSIGHT_AI_PORT_SOURCE_DIR"
source .nsight-ai-port/env.sh

nsys profile --sample=process-tree \
  -o "$PROFILE_ROOT/round_0/baseline" \
  --force-overwrite=true \
  bash "$P_RUN_PERFORMANCE"
```

Verify the report contains CPU sample / hotspot data. If empty or
errors mention sampling/perf/ptrace — **stop** (see above).

### Validate sample count (too-short runs)

Sampling can succeed but collect **zero samples** when the problem
run is too short (e.g. ~10 ms). This is **not** an environment
failure — scale the workload.

After export (see below), check CPU sample count. Example:

```bash
REPORT="$PROFILE_ROOT/round_0/baseline.nsys-rep"
SQLITE="${REPORT%.nsys-rep}.sqlite"

# Materialize SQLite if needed
nsys stats --force-export=true "$REPORT" > /dev/null

SAMPLE_COUNT=$(python3 - "$SQLITE" <<'PY'
import sqlite3, sys
db = sys.argv[1]
try:
    c = sqlite3.connect(db)
    # Table/column names vary — inspect schema if this fails
    n = c.execute(
        "SELECT COUNT(*) FROM SAMPLING_CALLCHAINS WHERE stackDepth=0"
    ).fetchone()[0]
    print(n)
except Exception:
    print(0)
PY
)
echo "CPU samples (stackDepth=0): $SAMPLE_COUNT"
```

**If sample count is zero or very low** (heuristic: problem run
**under ~1 s wall time** or **fewer than ~100 samples**):

1. **Stop** — do not add NVTX or plan from an empty trace.
2. Tell the user the run was too short to profile.
3. Offer to **scale the workload** (larger problem size, more
   timesteps, repeat the compute loop) and update
   the active performance run script / inputs — **with user approval**.
4. Re-capture Round 0 after scaling.
5. Record wall time, sample count, and any scaling decision in
   `extraction_meta.txt`.

Use hotspots to choose where to add **`function:`** NVTX tags (see
`flavors/{from_lang}/nvtx.md` naming grammar). Then commit NVTX separately.

## Post-NVTX and Post-Port Profiles

```bash
nsys profile \
  -o "$PROFILE_ROOT/round_N/instrumented" \
  --force-overwrite=true \
  bash "$P_RUN_PERFORMANCE"
```

GPU build (after porting). `USE_{FRAMEWORK}` is the upper-cased
`config.json:framework` (e.g. `USE_OPENACC`, `USE_CUDA`):

```bash
source .nsight-ai-port/env.sh
USE_{FRAMEWORK}=1 bash "$P_BUILD_PERFORMANCE"
nsys profile \
  -o "$PROFILE_ROOT/round_N/gpu" \
  --force-overwrite=true \
  env USE_{FRAMEWORK}=1 bash "$P_RUN_PERFORMANCE"
```

Write `gaps.txt` after NVTX runs — large unmarked gaps → refine tags
before planning.

## Canonical Stats Export (Required)

**Always** pass `--force-export=true` when running `nsys stats`.
Without it, nsys may reuse a **stale SQLite cache** and return empty or
outdated reports.

### Choose top K (agent discretion)

Pick **K** based on codebase scale and report size — there is no fixed
default. Starting heuristics:

| Codebase scale | Starting K |
|----------------|------------|
| Toy / ≤ 20 source files | 10–15 |
| Medium / tens of files | 20–40 |
| Large / hundreds+ files | 40–80 |

You may **raise K and re-run `nsys stats` only** (no new profile
capture) if truncated exports omit likely hotspots. Record every
decision in `extraction_meta.txt`.

### Export commands

Replace `REPORT`, `ROUND_DIR`, and `K` below.

```bash
REPORT=.nsight-ai-port/profiles/round_0/baseline.nsys-rep
ROUND_DIR=.nsight-ai-port/profiles/round_0
K=25   # agent-chosen; adjust and re-export if needed

nsys stats --report nvtx_sum --format csv \
  --force-export=true --force-overwrite=true \
  "$REPORT" > "$ROUND_DIR/nvtx_sum.csv"

# CPU hotspots (round 0 / pre-NVTX) — report names vary by nsys version.
# Try stats CSV reports first:
for RPT in cpu_sampling_sum osrt_sum; do
  nsys stats --report "$RPT" --format csv \
    --force-export=true --force-overwrite=true \
    "$REPORT" > "$ROUND_DIR/cpu_hotspots_raw.csv" 2>/dev/null \
    && break
done

# If CSV is empty or the report does not exist (e.g. nsys 2026.x has no
# cpu_sampling_sum), query SAMPLING_CALLCHAINS via python3 — do not require
# the sqlite3 CLI:
if [ ! -s "$ROUND_DIR/cpu_hotspots_raw.csv" ]; then
  nsys stats --force-export=true "$REPORT" > /dev/null
  SQLITE="${REPORT%.nsys-rep}.sqlite"
  python3 - "$SQLITE" "$K" <<'PY' > "$ROUND_DIR/cpu_hotspots_raw.csv"
import csv, sqlite3, sys

db, k = sys.argv[1], int(sys.argv[2])
conn = sqlite3.connect(db)
# Leaf frames only; column names vary — inspect .schema if needed
rows = conn.execute("""
    SELECT symbol, SUM(samples) AS samples
    FROM SAMPLING_CALLCHAINS
    WHERE stackDepth = 0
    GROUP BY symbol
    ORDER BY samples DESC
    LIMIT ?
""", (k,)).fetchall()
w = csv.writer(sys.stdout)
w.writerow(["symbol", "samples"])
w.writerows(rows)
PY
fi
```

Sort by sample count or inclusive/total time **descending**, keep top
**K** rows, write `$ROUND_DIR/cpu_hotspots.csv` (or trim `nvtx_sum.csv`
similarly).

Write **`extraction_meta.txt`** (required):

```text
report: .nsight-ai-port/profiles/round_0/baseline.nsys-rep
top_k: 25
total_rows: 847
rows_exported: 25
sample_count: 897
problem_wall_time_s: 1.19
time_coverage_pct: 96.8
dropped_time_pct: 3.2
extraction_method: SAMPLING_CALLCHAINS/python3
notes: Scaled N=256->2048 after zero samples; re-exported with K=40
```

**Never** load `.nsys-rep` into LLM context. Feed planning only:
`summary.txt`, `gaps.txt`, `extraction_meta.txt`, and trimmed CSVs.

### Optional SQLite / NVTX queries

If CSV reports are unwieldy for **NVTX** data, export once with
`--force-export=true`, then query the generated `.sqlite` (path shown
in nsys stats output). Prefer **python3** over the `sqlite3` CLI —
the CLI is often absent in minimal environments.

Example pattern (table/column names vary by nsys version — inspect
schema):

```sql
-- Illustrative: NVTX ranges sorted by time descending, limit K
SELECT name, total_time, percent
FROM NVTX_EVENTS
ORDER BY total_time DESC
LIMIT 25;
```

Log the query, K, and extraction method in `extraction_meta.txt`.

## Windowing Guidance

Problem runs may loop thousands of times. Inspect **one**
`iter:*` iteration — not the full timeline.

Write `summary.txt` with:

- Problem window (exclude init/teardown ~5% each end if needed)
- One iteration's NVTX hierarchy (`iter:`, `phase:compute:`,
  `phase:host:`, `function:`, `block:`) with inclusive time %
- Gap summary
- Hot symbols mapped to `file:function` (see planning-from-trace)

## Performance Assessment (post-translate)

Run this at **Phase 5** of every round, after the GPU build and a fresh
GPU profile (`profiles/round_N/gpu.nsys-rep`). The goal is a plain
answer for the user: **how much of the timeline is GPU work, and how
much whitespace (idle/host-bound time) remains.**

### Metrics to compute

| Metric | Meaning |
|--------|---------|
| `gpu_active_pct` | GPU busy time (kernels + memops, **union** — no double-count) as a % of the problem window |
| `whitespace_pct` | `100 - gpu_active_pct` — host-bound / idle time left in the window |
| per-hot-region | each Phase-1 hot `function:` — now GPU (has kernels) or still host |
| round delta | `gpu_active_pct` and problem wall time vs the previous round |

### Window

Use the **median `iter:` range** by duration (same window `summary.txt`
uses). Three rules make this measurement trustworthy:

- **Only `iter:` ranges are candidates.** Never `phase:`. A
  `phase:host:` section such as init (CUDA context creation + managed
  allocation) is routinely longer than any single iteration and
  contains **no kernels** — selecting it reports `gpu_active_pct: 0.0`
  for a perfectly good GPU build.
- **Median, not longest.** Iteration 1 absorbs one-time costs (lazy
  context init, JIT, first-touch page migration under managed memory),
  so the longest iteration is the least representative one.
- **Fallbacks, in order:** no `iter:` ranges → the union span of
  `phase:compute:` ranges; none of those → the full GPU span.

Exclude `phase:host:*` from "expected host" commentary so remaining
whitespace reads as real opportunity. Classing host work in the tag
(rather than matching known names like `io`/`mpi_*`) is what makes the
exclusion complete.

**Sanity check:** if the chosen window contains **zero** kernels in a
GPU build, that is a windowing or tagging failure — not a 0% result.
Re-check the NVTX markers and re-profile before reporting any number.

### Compute (python3 + SQLite)

Export the GPU report once, then compute the union of GPU-busy
intervals (overlapping streams must not be double-counted). Table and
column names vary by nsys version — inspect `.schema` if a query fails.

```bash
REPORT=.nsight-ai-port/profiles/round_N/gpu.nsys-rep
ROUND_DIR=.nsight-ai-port/profiles/round_N
nsys stats --force-export=true "$REPORT" > /dev/null
SQLITE="${REPORT%.nsys-rep}.sqlite"

python3 - "$SQLITE" <<'PY' > "$ROUND_DIR/perf_summary.txt"
import sqlite3, sys

db = sys.argv[1]
c = sqlite3.connect(db)

def intervals(table):
    try:
        return c.execute(f"SELECT start, end FROM {table}").fetchall()
    except Exception:
        return []

busy = []
for t in ("CUPTI_ACTIVITY_KIND_KERNEL",
          "CUPTI_ACTIVITY_KIND_MEMCPY",
          "CUPTI_ACTIVITY_KIND_MEMSET"):
    busy += intervals(t)

# Window: the MEDIAN `iter:` range by duration.
#   - Only `iter:` is a candidate. `phase:host:init` (context creation +
#     managed alloc) outlasts any iteration and holds no kernels, so
#     admitting `phase:%` here reports gpu_active_pct 0.0 on a good build.
#   - Median, not longest: iteration 1 absorbs lazy context init, JIT and
#     first-touch page migration, so it is the least representative.
def _ranges(pattern):
    try:
        rows = c.execute(
            "SELECT start, end FROM NVTX_EVENTS WHERE text LIKE ?"
            " ORDER BY (end - start)", (pattern,)
        ).fetchall()
        return [r for r in rows if r[0] is not None and r[1] is not None]
    except Exception:
        return []

window = None
iters = _ranges("iter:%")
if iters:
    window = iters[len(iters) // 2]          # median iteration
else:
    compute = _ranges("phase:compute:%")     # no repeating unit
    if compute:
        window = (min(s for s, _ in compute), max(e for _, e in compute))
if window is None and busy:
    window = (min(s for s, _ in busy), max(e for _, e in busy))

# Clip busy intervals to the chosen window BEFORE unioning, so GPU-busy
# time outside the window (warmup, other iterations) cannot push the
# percentage above 100%.
union_ns, window_ns = 0, 0
if window is not None:
    w_s, w_e = window
    window_ns = w_e - w_s
    clipped = sorted(
        (max(s, w_s), min(e, w_e)) for s, e in busy if min(e, w_e) > max(s, w_s)
    )
    cur_s, cur_e = None, None
    for s, e in clipped:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                union_ns += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        union_ns += cur_e - cur_s

gpu_pct = 100.0 * union_ns / window_ns if window_ns else 0.0
print(f"window_source: {'iter' if iters else 'phase:compute/fallback'}")
print(f"iter_ranges: {len(iters)}")
print(f"gpu_busy_ns: {union_ns}")
print(f"window_ns: {window_ns}")
print(f"gpu_active_pct: {gpu_pct:.1f}")
print(f"whitespace_pct: {100.0 - gpu_pct:.1f}")
if union_ns == 0:
    print("WINDOW_SUSPECT: no kernels in the chosen window — treat this as"
          " a tagging/windowing failure, NOT a 0% measurement.")
PY
cat "$ROUND_DIR/perf_summary.txt"
```

Per-region status: cross-reference each hot `function:` NVTX range
against kernels launched inside its time span (or use
`nsys stats --report cuda_gpu_kern_sum`/`gpukernsum` — name varies).
A hot region with no kernels in the GPU build is **still on host** and
is the prime candidate for the next round.

Write `gpu_active_pct` / `whitespace_pct` into the round record
(`progress.json:rounds[]`) and feed them to the iteration report. Never
load `.nsys-rep` into LLM context — feed only `perf_summary.txt`,
`summary.txt`, `gaps.txt`, and trimmed CSVs.

## Viewing Timelines (User)

```bash
nsys-ui .nsight-ai-port/profiles/round_N/instrumented.nsys-rep
```

Binary reports stay local (gitignored).

## Convergence Heuristic (Reporting Only)

User decides whether to run another round. See iteration report
template.
