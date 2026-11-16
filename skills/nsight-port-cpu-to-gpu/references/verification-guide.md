# Verification Guide

Verification is the correctness gate for every phase. This guide
covers the verification patterns, when to use each, and how to
diagnose failures.

## Scripts

The [setup guide](setup-guide.md) owns script naming, location, and
ownership rules. This guide uses the correctness scripts.

## Flavor Placeholders

Substitute placeholders in commands and examples below (naming
convention in [SKILL.md](../SKILL.md#flavor-packs)):

| Placeholder | Source | Examples |
|-------------|--------|----------|
| `USE_{FRAMEWORK}` | build env var, from `config.json:framework` | `USE_OPENACC`, `USE_CUDA` |
| `{include_guard}` | **`config.json:include_guard`** | usually `USE_{FRAMEWORK}` |

`include_guard` is always read from `config.json`, never constructed:
Q7 may have overridden its `USE_{FRAMEWORK}` default.

## Verification Levels

Different phases require different levels of verification.
Never run more than needed, but never skip what's required.

### Environment Setup

**Always** source `.nsight-ai-port/env.sh` before running
verification scripts. This sets static paths
(`NSIGHT_AI_PORT_SOURCE_DIR`, `NSIGHT_AI_PORT_BUILD_DIR`).

Build-mode flags (`USE_{FRAMEWORK}`) are passed
**directly** to the build/run command, NOT to `source env.sh`. This is
because bash unwinds prefix variables after the `source` builtin
returns — so `USE_{FRAMEWORK}=1 source env.sh` does not propagate the
variable to subsequent commands.

```bash
# CPU: source env then run
source .nsight-ai-port/env.sh && bash .nsight-ai-port/build_correctness.sh

# GPU: pass USE_{FRAMEWORK} directly to the command
source .nsight-ai-port/env.sh && \
  USE_{FRAMEWORK}=1 bash .nsight-ai-port/build_correctness.sh
```

### Compiler Consistency (CPU vs GPU)

Use the **same Fortran compiler family** for CPU and GPU builds
when the project already targets that vendor (e.g. nvfortran for
both paths). Mixing gfortran on CPU and nvfortran on GPU breaks
`#ifdef` call sites, module ABI, and preprocessor behavior.

If the CPU path currently uses a different compiler, migrate CPU
flags to the GPU vendor **before** adding `#ifdef` dispatch at call
sites. Record the chosen compiler in `.nsight-ai-port/build_notes.txt`.

Once `#ifdef` appears in a `.f90` source, **every** Fortran compile line
that touches that file needs preprocessing enabled — `-Mpreprocess` for
nvfortran, `-cpp`/`-fpp` for gfortran, `-fpp` for ifx — including
configurations that never define the guard. A pure-Fortran CPU baseline
built by gfortran stops building entirely otherwise. Settle it during build
exploration and verify by building every configuration; see the active
flavor's `translation.md` (`flavors/{to_lang}/{framework}/`, e.g.
`flavors/fortran/openacc/` or `flavors/fortran/openmp/`).

### The build system owns compiler choice

Write compiler decisions into the build files, not into a prefix of
variables typed on each command. One environment variable —
`USE_{FRAMEWORK}` — selects CPU or GPU; everything else (compiler paths,
flags, object locations) is decided in the Makefile or CMakeLists, where
it is version-controlled and applied the same way every time.

In Make, that means the assignment operator matters. Measured with GNU
Make 4.x:

| Assignment in Makefile | vs ambient env var | vs `make VAR=…` | vs `make -e` |
|------------------------|--------------------|-----------------|--------------|
| `VAR ?= x` | **env wins** | cmdline wins | env wins |
| `VAR := x` | **Makefile wins** | cmdline wins | env wins |
| `override VAR := x` | Makefile wins | **Makefile wins** | **Makefile wins** |

So use `:=` for `CC`/`CXX`/`FC`/`NVCC`: it defends against a stale
compiler variable inherited from the shell while still letting a human
say `make CXX=clang++`. Keep `?=` for `USE_{FRAMEWORK}` itself, which
*must* be settable from the environment. Reach for `override :=` only
when the project's own wrapper scripts pass compilers on the make command
line. Never invoke `make -e`, which promotes every ambient variable above
the Makefile.

Record the **ambient** `CC`/`CXX`/`FC`/`NVCC` values in
`.nsight-ai-port/build_notes.txt` even when the Makefile pins its own — if a build later
behaves differently elsewhere, the inherited environment is the first
thing to compare, and it cannot be recovered after the fact.

### Per-configuration build trees

CPU and GPU builds must never share an object tree or a CMake binary
directory:

```bash
$NSIGHT_AI_PORT_BUILD_DIR/cpu/    # objects + binary, no USE_{FRAMEWORK}
$NSIGHT_AI_PORT_BUILD_DIR/gpu/    # objects + binary, USE_{FRAMEWORK}=1
```

Two CMake configurations sharing one binary dir means the second silently
reuses the first one's cache. Two Make configurations sharing one object
tree means the linked binary can be part CPU and part GPU — and the
symptoms of that (CPU and GPU results identical, no kernels in the
profile) look exactly like translation bugs, so it costs a lot of
debugging in the wrong place.

This is also what makes `make clean` unnecessary rather than merely
discouraged.

### Never clean *unconditionally*

The rule is never clean unconditionally — not "never clean". A blanket
`make clean` in a build script destroys incremental builds and makes
every round slow; a build that quietly reuses objects compiled with
different flags is worse. Separate trees per configuration remove nearly
every reason to clean.

When a full rebuild really is needed, it is for one of these, and you
should say which:

- A semantics-changing `#define` reached only some translation units.
- Stale Fortran `.mod` files — module interface changes are a notorious
  blind spot in `make` dependency tracking.
- The compiler itself changed.
- CMake: the source structure changed and `CMakeCache.txt` still holds
  the old file list.

Prefer deleting only the affected configuration's tree over a
project-wide clean, and record the reason.

### Verify the build did what you asked

Exit code 0 does not mean the build did what you intended. After the
first build of each configuration, confirm:

```bash
# Which compiler actually ran, and did the guard reach it?
make -n 2>&1 | grep -E 'nvcc|nvc\+\+|nvfortran|gfortran|g\+\+' | head
make -n 2>&1 | grep -c -- -DUSE_{FRAMEWORK}    # expect > 0 on the GPU path

# Did objects land in the configuration-specific tree?
find "$NSIGHT_AI_PORT_BUILD_DIR" -name '*.o' -newermt '-5 minutes' | head
```

Cheap checks, and they catch the common case where a stale environment
variable or an unwired subdirectory kept the guard from reaching the
compiler.

### Build-only

**When:** After scaffold creation, after adding each OpenACC
directive chunk.

```bash
source .nsight-ai-port/env.sh && \
  USE_{FRAMEWORK}=1 bash .nsight-ai-port/build_correctness.sh
```

Confirms the code compiles with GPU flags.

### Build + Standalone Test

**When:** After scaffold, after each accelerator chunk, after
numerical verification.

```bash
source .nsight-ai-port/env.sh && \
  USE_{FRAMEWORK}=1 bash .nsight-ai-port/build_correctness.sh
./test_{function}
```

Confirms the code compiles and the test passes.

### Build + Standalone Test + Compute-Sanitizer

**When:** After numerical verification (Phase 4d).

```bash
source .nsight-ai-port/env.sh && \
  USE_{FRAMEWORK}=1 bash .nsight-ai-port/build_correctness.sh
compute-sanitizer ./test_{function}
```

`compute-sanitizer` detects GPU memory errors (out-of-bounds,
uninitialized reads, race conditions). If not available, skip
this and use a regular test run.

### Dual-Path (Full Integration Verification)

**When:** After integration (Phase 4e). This is the most
thorough check.

Five steps, all must pass:

```text
Step 1: CPU Build
  source .nsight-ai-port/env.sh && bash .nsight-ai-port/build_correctness.sh

Step 2: CPU Run
  source .nsight-ai-port/env.sh && bash .nsight-ai-port/run_correctness.sh

Step 3: GPU Build
  source .nsight-ai-port/env.sh && \
    USE_{FRAMEWORK}=1 bash .nsight-ai-port/build_correctness.sh

Step 4: GPU Run
  source .nsight-ai-port/env.sh && \
    USE_{FRAMEWORK}=1 bash .nsight-ai-port/run_correctness.sh
  # Check: stdout MUST contain
  # "NSIGHT_AI_PORT_FRAMEWORK_EXEC: {function_name}"
  # If missing, this is a FAILURE even if exit code is 0.

Step 5: Standalone Test (sanitized)
  source .nsight-ai-port/env.sh && \
    USE_{FRAMEWORK}=1 bash .nsight-ai-port/build_correctness.sh
  compute-sanitizer ./test_{function}
```

**Why five steps?**

- Steps 1-2 catch **CPU regressions** — the `#ifdef` blocks
  must not break the non-GPU build path. Common failures:
  unguarded `USE translated_mod`, missing `#endif`, syntax
  errors in the `#else` path.

- Steps 3-4 confirm the **GPU path works end-to-end**. The
  execution marker check (Step 4) proves the `#ifdef` dispatch
  actually ran the GPU code — without it, you might have a
  working build that silently runs the CPU path.

- Step 5 catches **GPU memory errors** that don't cause
  visible test failures but indicate correctness problems.

## Standalone Test Harness

### Test `main` must be in the original language

Every standalone test executable's `main` / `program` unit must be in the
**original** language — the language the project was written in before any
translation. For a Fortran project ported to CUDA C++, every test
executable is a Fortran program, in Stage A and Stage B alike. For a C++
project, every test executable has a C++ `main`.

This is not a style preference. The translated function is reached from the
original code through an interop shim — a `bind(C)` interface, a
`type(c_ptr)` argument, an `extern "C"` declaration — and **that shim is
part of what you are shipping**. A test whose `main` is in the target
language calls the candidate natively and never exercises the shim, so the
whole class of interop defects survives a green test suite: argument
copies at the `bind(C)` boundary, descriptor-versus-address mismatches,
name mangling, wrong `value` attributes, runtime-library link order.

The rule follows directly:

| Original language | Test `main` | Reference call | Candidate call |
|---|---|---|---|
| Fortran | Fortran `program` | the original directly (`use` its module) | via `bind(C)` interface — the production shim |
| C/C++ | C++ `main` | the original directly | via `extern "C"` declaration |

Calling the reference **directly** is the other half of it. Because the
harness is already in the original language, the reference needs no wrapper
at all, which removes a shim that existed only for the test and could
itself disagree with the real call path.

Build implication: the mixed-language link is driven from the original
language's compiler (e.g. `nvfortran ... -c++libs` for a Fortran program
linking C++ objects). Record that in `.nsight-ai-port/build_notes.txt` during build
exploration.

## GPU Execution Marker

**One marker per translated function, printed once, from inside the
translated function.** Not at the call site, and not one per loop nest.
Putting it inside the body is what makes it evidence: the marker fires if
and only if the translated code actually executed.

```fortran
subroutine compute_kernel_gpu(args)
  logical, save :: marker_printed = .false.
  ...
  if (.not. marker_printed) then
    print *, 'NSIGHT_AI_PORT_FRAMEWORK_EXEC: compute_kernel'
    marker_printed = .true.
  end if
```

The C/C++ form is the same shape with a function-local
`static bool marker_printed = false;`.

Where the body lives decides whether a guard is needed:

| Translation mode | Marker goes in | Guard around it |
|---|---|---|
| `sibling` | the translated `{function}_{framework}` body | none — the file only compiles on the GPU path |
| `inline` | the `#ifdef {include_guard}` region inside the edited function | already inside the guard |
| Cross-language Stage A | the `{function}_{to_lang}` dispatcher (`NSIGHT_AI_PORT_LANG_EXEC`) | none |
| Cross-language Stage B | the `{function}_{framework}` body (`NSIGHT_AI_PORT_FRAMEWORK_EXEC`) | none |

For MPI programs, guard the print on rank 0 so the marker appears once
rather than once per rank:

```fortran
  if (rank == 0 .and. .not. marker_printed) then
    print *, 'NSIGHT_AI_PORT_FRAMEWORK_EXEC: compute_kernel'
    marker_printed = .true.
  end if
```

The rank must be obtained the way the surrounding code already does it —
a stored module variable, an `MPI_Comm_rank` call, or the project's own
accessor. If the translated body has no access to a rank, print
unconditionally rather than inventing an MPI dependency; duplicate marker
lines are harmless to the grep.

**Verification check:** After GPU run, grep stdout for the
marker string. **A GPU run that exits 0 but lacks the marker
is a FAILURE, not a pass.** This is the primary defense against
silent CPU-path fallthrough — where the build succeeds, the
run succeeds, and numerical checks show 0 diff, but the GPU
code never actually ran.

**The marker stays on stdout**, including for programs whose scientific
output goes to stdout. It is status, not an error, and moving it to stderr
would hide it from the ordinary run log. The consequence is that a
correctness gate must never diff raw stdout against a golden file — it must
extract the data lines and compare those numerically. See
[Golden references are data files](setup-guide.md#golden-references-are-data-files-not-stdout).

If the marker is missing:
- The `#ifdef` dispatch at the call site was not taken, so the translated
  body never ran (check that `{include_guard}` is defined during
  compilation — was it passed to the build?)
- The binary is stale — rebuild the **affected configuration tree**
  (e.g. `$NSIGHT_AI_PORT_BUILD_DIR/gpu/`), not the whole project; see
  [never clean unconditionally](#never-clean-unconditionally)
- The function is not being called (check the test harness)
- `USE_{FRAMEWORK}` was not propagated to the build command (check
  that it's passed as `USE_{FRAMEWORK}=1 bash .nsight-ai-port/build_correctness.sh`,
  not via `source env.sh`)

## Environment Variables

**Static (set by `env.sh`):** `NSIGHT_AI_PORT_SOURCE_DIR`,
`NSIGHT_AI_PORT_BUILD_DIR`, and any library paths. These are
set once by sourcing `env.sh` and persist for the session.

**Build-mode (passed directly to commands):** These MUST be
passed as prefix variables to
`bash .nsight-ai-port/build_correctness.sh` /
`bash .nsight-ai-port/run_correctness.sh`, never set in
`env.sh`:

| Variable | Value | How to pass |
|----------|-------|-------------|
| `USE_{FRAMEWORK}` | `1` | `USE_{FRAMEWORK}=1 bash .nsight-ai-port/build_correctness.sh` |
| `NV_ACC_TIME` | `1` | `NV_ACC_TIME=1 bash <active-perf-run-script>` |
| `ABS_TOL` | e.g. `1e-10` | `ABS_TOL=1e-10 bash .nsight-ai-port/run_correctness.sh` |
| `REL_TOL` | e.g. `1e-8` | `REL_TOL=1e-8 bash .nsight-ai-port/run_correctness.sh` |
| `WMAD_TOL` | e.g. `1e-12` | `WMAD_TOL=1e-12 bash .nsight-ai-port/run_correctness.sh` |
| `REL_FLOOR` | e.g. `1e-10` | `REL_FLOOR=1e-10 bash .nsight-ai-port/run_correctness.sh` |
| `MAX_REL_CHECK` | `on` (default) / `off` | `MAX_REL_CHECK=off bash .nsight-ai-port/run_correctness.sh` |

See [Comparing Numerical Results](#comparing-numerical-results) for what
`WMAD_TOL`, `REL_FLOOR`, and `MAX_REL_CHECK` mean and when waiving the
outlier gate is defensible.

For `NV_ACC_TIME=1`, you may also need to set the CUPTI library
path:

```bash
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/.../compilers/lib:$LD_LIBRARY_PATH
```

Find the CUPTI path with:
```bash
find /opt/nvidia -name "libcupti.so" 2>/dev/null
```

## Comparing Numerical Results

### The reference is always the original language

Whatever the round produced — a CPU translation, a GPU translation, or
both — the implementation it must agree with is the **original code in
the original language**. That is the only source of truth. When a CPU
translation and a GPU translation disagree with each other, it matters
only because at least one of them disagrees with the original, and
comparing each against the original tells you which. Report the
CPU-versus-GPU difference for diagnosis; never make it the gate.

### A difference of exactly zero deserves a second look

When a GPU-versus-CPU comparison reports a max difference of exactly
`0.0`, first rule out that both paths are running the *same* code. The
usual cause is a "CPU reference" that is actually a dispatcher: built with
the framework guard on, it forwards to the GPU, so the test compares the
GPU against itself and passes while verifying nothing. Check that the
reference symbol's body contains no framework `#ifdef` (each flavor's
Stage B guide names the symbol to use).

Bit-exact agreement is genuinely possible — an elementwise kernel with no
reductions and unchanged operation order can match exactly. But reductions
reassociate on the GPU, so for anything that sums or reduces, exact zero is
better evidence of self-comparison than of correctness. Once you have
confirmed which case you are in, say which one in the round's notes.

### Never gate on a global integral

A bare sum — or a mean, or a total-energy checksum — collapses the whole
field into one number, and it fails in both directions:

- **It masks cell-wise error.** Errors of opposite sign cancel in the
  integral, so a field can be wrong in every cell and still match the
  reference to roundoff. The smoother the field, the more completely they
  cancel, and smooth fields are the ordinary case. A passing integral is
  not evidence.
- **It is ill-conditioned.** When a field has mixed signs and a near-zero
  mean (a velocity field, a perturbation, an anomaly), the integral is a
  catastrophic cancellation: its *relative* error can be 1e-3 while every
  cell agrees to 1e-15. A failing integral is not evidence either.

Compare the **field**, not a scalar summary of it. Global integrals are
still worth printing as diagnostics — a conservation check is meaningful
on its own terms — but they never gate.

### The metric: weighted mean absolute difference

Gate on a normalized, weighted mean absolute difference against the
reference field `ref`:

```text
wmad = sum_i( w_i * |a_i - ref_i| ) / sum_i( w_i * |ref_i| )
```

`w_i` is the physical weight of element `i` — cell volume or area on a
non-uniform mesh, otherwise 1. Dividing by `sum(w*|ref|)` (absolute
values, so nothing cancels) makes `wmad` dimensionless and scale-free,
which means one tolerance works across fields and problem sizes.

Alongside it, track the **worst element and how many elements are bad**.

### `compare_field` is shipped source, not something you write

The skill ships one compiled, self-tested implementation per language under
`assets/test-harness/`. Copy it in and call it — do not hand-write the
metric into a test, and do not paste it into a document. Which file you
copy is decided by `config.json:from_lang`, the **original** language,
because that is the language every test `main` is written in. It is never
decided by the target language or the framework:

| `from_lang` | File to copy |
|---|---|
| `fortran` | `assets/test-harness/compare_field.f90` (module `nsight_port_compare`) |
| `cpp` | `assets/test-harness/compare_field.hpp` (namespace `nsight_ai_port`) |

#### One copy per port, at the source root

The destination is `{source_dir}/{framework}/test/` for a
same-language port and `{source_dir}/{to_lang}/test/` for a
cross-language one. Either is fixed for the whole port, so this is
**one copy for the whole project** — not one per source directory,
and not one per function.

In a cross-language port Stage A owns the copy, because the harness
language follows `from_lang` and Stage A is the first stage to need
it; Stage B reuses that path and copies nothing. Example: `fortran`
→ `cpp` → `cuda` copies the Fortran harness under `cpp/test/`, never
under `cuda/test/`.

This is the single exception to the sibling-directory rule. Every other
artifact lives next to the original it was translated from, because it
*is* a translation of that file. The harness is not: it is the same
bytes for every function in the port, so a copy next to each original
would be N copies of one file.

For the Fortran harness that matters beyond tidiness. Every copy
declares the same module, `nsight_port_compare`, and the documented
build compiles all Fortran into one shared module directory
(`-module ${MODDIR} -I${MODDIR}`). N copies therefore means N compiles
writing the same `nsight_port_compare.mod` into one directory: a race
under `make -j`, and last-writer-wins if the copies ever drift apart.
One copy has neither problem. The C++ header is header-only, so it
carries no link or module hazard — but N copies of it can still drift,
which is reason enough.

Three consequences:

- **A two-stage cross-language port copies it once, in Stage A.** Stage
  B keeps the original language's driver, so it reuses Stage A's harness
  and copies nothing. `from_lang` selects the file; the framework never
  does.
- **The Fortran harness is compiled once.** One object, linked by every
  test; the tests resolve the module through the `-I${MODDIR}` already
  on their compile line. Do not add a per-test copy of that rule.
- **The C++ harness is not compiled at all.** It is header-only, so a
  test needs only its directory on the include path.

Both harnesses expose the same three-call contract:

| Call | What it does |
|---|---|
| `tolerances_for` | element-kind defaults, then environment overrides |
| `compare_field(candidate, reference, …, tol [, weights])` | returns `wmad`, `max_rel`, `n_exceed`, `n_total`, the worst element, and a non-finite flag |
| `report_and_gate(result, tol, label)` | prints the four required numbers and applies the gate; returns true on pass |

`weights` carries the physical weight of each element — cell volume or
area on a non-uniform mesh — and is omitted on a uniform one.

The gate `report_and_gate` applies:

1. a shape disagreement between candidate, reference or weights fails
   unconditionally, and no metric is computed — it is a bug in the test,
   not a numerical result. Only the Fortran harness can check this; the
   C++ one is passed an explicit element count;
2. any non-finite value in either field fails unconditionally;
3. `wmad > WMAD_TOL` fails — this is the gate and is not discretionary;
4. `n_exceed > 0` fails **when `MAX_REL_CHECK` is on** — this one is
   reviewable, see [below](#reviewing-an-outlier-failure).

It prints all four numbers, including the exceedance count and the worst
element — a bare max is not reviewable:

```text
compare compute_kernel: wmad 3.10000e-14  max_rel 4.20000e-06
  n_exceed 3 of 500000  worst_idx 128041
```

Tolerances come from the environment, never hardcoded: `WMAD_TOL`
(default `1e-12` double / `1e-6` float), `ABS_TOL` and `REL_TOL` as
elsewhere in this guide, `REL_FLOOR` for the magnitude floor (default
`1e-10`), and `MAX_REL_CHECK` (default on) for the outlier gate. The
shipped implementation reads all five, so a test never has to.

### Reviewing an outlier failure

`wmad` passing while `n_exceed > 0` is a common and often benign outcome:
GPU reductions reorder, FMA contraction differs, and a handful of cells
whose value is near zero can show a large *relative* difference from
pure roundoff. A tiny-valued cell disagreeing is not proof of a bug.

You are expected to **review** such a failure rather than either
accepting it blindly or loosening tolerances until it disappears.
Investigate first, then apply the least drastic remedy that fits what you
found:

1. **Raise the magnitude floor** (`REL_FLOOR`) if the exceedances sit in
   cells orders of magnitude below the field peak. This keeps the check
   active where it is meaningful.
2. **Disable the outlier gate** (`MAX_REL_CHECK=off`) and rely on `wmad`
   alone, if the exceedances are genuinely inherent to the algorithm.

Either way, record in `progress.json` what you saw (`n_exceed`, the worst
index and its magnitude relative to the peak) and why you concluded it
was numerics rather than a defect.

**Never waive** — these are defects, not noise:

- Any non-finite value (NaN, Inf) anywhere in either field.
- `n_exceed` that grows with problem size, iteration count, or thread
  count. Roundoff does not scale that way; a race or an indexing bug
  does.
- Exceedances in cells at or near the field peak.
- Exceedances in a **structured** pattern — a contiguous run, a face, a
  stride, a boundary row. Structure means indexing, halo, or bounds, and
  its `wmad` may well still pass.
- A `wmad` failure. The outlier gate is discretionary; `wmad` is not.

### Scalar checks: relative tolerance does not fit every quantity

When you gate on a **scalar** the project already reports — a norm, a total,
a max error, a residual — the choice between relative and absolute
tolerance depends on how that scalar was computed. Using the wrong one
produces failures that no amount of retolerancing fixes sensibly.

| Kind | Examples | Gate with | Why |
|---|---|---|---|
| **Accumulated** | L2 norm, mass/energy total, mean, integral | **Relative** | Magnitude is set by the data; relative error is stable and scale-free |
| **Cancellation** | `max\|u - u_exact\|`, residuals, any difference of nearly-equal values | **Absolute**, floored at the operands' roundoff | The result's own magnitude carries no information about the error in it |

A cancellation quantity has already lost its significant digits by
construction. `max|u - u_exact|` of `4e-11` is the difference of two
numbers of order 1, so its accuracy is limited by roundoff on those
operands — about `1e-16` — not by `4e-11`. Recompute it with a different
compiler and the absolute value moves by `~1e-16`, which is a **relative**
change of `~2.5e-6`. Nothing is wrong, but a relative gate anywhere below
`1e-5` fails.

So for a cancellation quantity, set an absolute tolerance from the
**operands'** scale, not the result's:

```text
abs_tol  ~  eps * peak|operand| * modest_safety_factor
         ~  2.2e-16 * 1.0 * 100   ->  ~1e-14      (double, O(1) fields)
```

and gate `|value_new - value_ref| <= abs_tol`. Deciding which bucket a
scalar belongs in is a one-line judgement: *is this number a sum of
same-signed contributions, or a difference of things that nearly cancel?*

This is also a reason to prefer the field comparison above over any scalar.
`compare_field` sidesteps the question entirely — it never forms the
cancelling scalar in the first place.

### When cell-wise comparison is the wrong question

Some algorithms are legitimately not reproducible cell-by-cell: chaotic
or turbulent evolution integrated over many steps, iterative solvers
converging by different paths, anything whose intermediate state amplifies
roundoff exponentially. In that case a `wmad` failure may say nothing
about correctness.

Before giving up on cell-wise comparison, **shorten the window** — compare
after one step or a few, where divergence has not yet had time to grow.
That is almost always possible in a unit test and keeps a strong check.

Only if that is genuinely impossible, fall back to invariants the port
must still preserve, and state which you chose and why:

- conserved quantities (mass, momentum, energy) tracked over the run;
- statistical measures (mean, variance, higher moments, energy spectra);
- the divergence *rate* between the two paths, compared against the
  original code's own run-to-run variability under a perturbed input.

Establish the original code's baseline variability first — otherwise you
have no way to distinguish "chaotic" from "broken". Record the choice, the
baseline, and the threshold in `progress.json`; a weaker check adopted
silently is indistinguishable from no check.

## Diagnosing Build Failures

### Step 1: Read the error

The compiler error message usually points to the exact file
and line. Read it carefully.

### Step 2: Run diagnostic commands

```bash
# See what make is actually doing:
make -n

# Find where a symbol is defined:
grep -rn "subroutine compute_kernel" src/

# Check module compile order:
grep -rn "use solver_mod" src/ | grep -v "\.o:"

# Check if a file is being compiled with GPU flags:
make -n 2>&1 | grep solver
```

### Step 3: Common failure patterns

**"Module file not found"**
- A `USE` statement references a module that hasn't been
  compiled yet. Fix the Makefile dependency order.

**"Unresolved external symbol"**
- The translated file isn't being compiled or linked. Check
  that every `{parent}/openacc/*.o` is included in the
  archive/link step.

**"Syntax error" in `#ifdef` block**
- Unmatched `#ifdef`/`#endif` pairs.
- Fortran continuation lines inside `#ifdef` blocks.
- Missing `#endif` for an `#ifdef`.

**"Undefined variable" in GPU region**
- A variable used inside `!$acc parallel loop` is not visible
  on the device.
- If `memory_mode` is `separate`: add the variable to a hoisted
  data region or add `present()` / `copyin` at the appropriate
  scope (not per hot-call kernel).
- If `memory_mode` is `unified` or `managed`: do NOT add
  `copyin`/`copyout` — check that the array is a dummy
  argument or module data accessible under unified memory.
  Check for derived-type references inside the compute region
  (flatten to local pointers before it).

**Redundant `#ifdef` inside `{parent}/openacc/` file**
- Remove `#ifdef {INCLUDE_GUARD}` wrapping `!$acc` directives.
  GPU-only files in `{parent}/openacc/` are never built without
  `-acc`. Keep `#ifdef` at call sites in the original source only.

**Link error: duplicate symbol**
- The same module is compiled from both the original and
  translated files. Ensure the translated module has a
  different name (e.g., `compute_kernel_openacc_mod`).

### Step 4: Fix and re-verify

Make the fix, then re-run the verification. If three fixes
fail, stop and ask the user.

## Diagnosing Test Failures

### Wrong numerical results

0. Check what actually failed: `wmad`, or only the outlier bound? See
   [Comparing Numerical Results](#comparing-numerical-results). A
   `wmad` failure is a real disagreement; an outlier-only failure with a
   small `n_exceed` may be numerics, and is reviewable.
1. Print intermediate values from both CPU and GPU paths.
2. Check whether the exceedances are **scattered** (consistent with
   roundoff) or **structured** — a contiguous run, a stride, a boundary
   face. Structure means an indexing, halo, or bounds bug.
3. Confirm you are not comparing a path against itself: an exact `0.0`
   difference on a GPU-vs-CPU comparison usually means the "CPU"
   reference is a dispatcher that also went to the GPU.
4. Large diffs: check memory mode first.
   - `separate`: missing `copyin` at hoisted scope means the
     GPU operates on uninitialized data.
   - `unified`/`managed`: spurious `copyin`/`copyout` or
     `!$acc data` regions may cause stale data or unnecessary
     transfers — remove forbidden clauses and re-test.

### Segfault or illegal memory access

Run with `compute-sanitizer` for detailed diagnostics:
```bash
compute-sanitizer --tool memcheck ./test_{function}
```

Common causes:
- Array out-of-bounds in GPU code
- Using a host pointer on the device
- Derived type referenced inside a compute region (flatten to local
  pointers before it)
- If `memory_mode` is `separate`: allocatable array not on the
  GPU (hoisted `!$acc data` / `enter data` missing)

### Test hangs

- Possible deadlock from mixing MPI and GPU operations
- Bound the run: `timeout "$VERIFY_TIMEOUT" ./test_{function}`, taking the
  budget in seconds from `verify_timeout` in `config.json` (default 600).
  `timeout` exits 124 on expiry, which is distinguishable from a genuine
  test failure
- Check if the test is waiting for user input

## Rollback Protocol

If verification fails catastrophically and you can't fix it:

Reversal changes files and history. Before changing either, show the user
the exact affected paths and commits (`git status --short` and
`git log --oneline "$GOOD"..HEAD`), save a recovery patch, and get explicit
approval at every engagement level.

1. **Read the rollback SHA first.** The working-tree
   `progress.json` is the only copy holding the latest SHA (a
   commit cannot contain its own hash), so read it before any
   reversal:
   ```bash
   GOOD=$(python3 -c "import json; print(json.load(open(
     '.nsight-ai-port/progress.json'))['last_working_commit'])")
   ```

2. **Validate the rollback boundary's freshness.** Stop without changing
   files or history unless `GOOD` is a nonempty commit and an ancestor of
   `HEAD`:
   ```bash
   git rev-parse --verify "$GOOD^{commit}"
   git merge-base --is-ancestor "$GOOD" HEAD
   ```
   List every other recorded verification checkpoint before deciding:
   ```bash
   python3 -c "import json
   p = json.load(open('.nsight-ai-port/progress.json'))
   shas = [f.get('commit_hash') for f in p.get('functions', [])]
   shas += [q.get('last_working_commit') for q in p.get('problems', [])]
   print('\n'.join(sorted({s for s in shas if isinstance(s, str) and s})))"
   git log --oneline "$GOOD"..HEAD
   git diff --stat "$GOOD"..HEAD
   ```
   Validate that each emitted checkpoint exists and is an ancestor of
   `HEAD`. If any unequal checkpoint descends from `GOOD`, then `GOOD` is
   stale: use the newest independently verified checkpoint only after
   reconciling `progress.json`, or stop and ask the user. Also stop when a
   later commit's verification or ownership is ambiguous. Show the user
   the log and diff summary and get explicit approval for this exact
   boundary before proceeding.

3. **Preserve recovery evidence.** Save `git diff --binary` under
   `.nsight-ai-port/recovery/` and inventory untracked files. Stop if a
   changed path predates the failed attempt or belongs to the user.

4. **Set uncommitted work aside first, only by explicit path.** `git revert`
   refuses to run while local edits touch the files it reverts, so clean the
   reviewed agent-owned paths before step 5. Restore each tracked path to
   `HEAD` (not `$GOOD`; step 5 carries it the rest of the way; the step 3
   patch keeps its edits), and move each agent-owned untracked file into
   the recovery directory:
   ```bash
   git restore --source=HEAD --staged --worktree -- path/to/agent-owned-file
   mkdir -p .nsight-ai-port/recovery/untracked
   mv -n -- path/to/agent-created-file .nsight-ai-port/recovery/untracked/
   ```
   Name every path individually; never act on a directory, a wildcard, or
   the whole working tree. Leave user-owned changes in place; if one touches
   a file a commit in step 5 would revert, stop and ask the user.

5. **Reverse committed work without rewriting history.** Revert only the
   approved agent-created commits in `"$GOOD"..HEAD`, newest first, one at a
   time:
   ```bash
   git revert --no-edit <newest-approved-sha>
   ```
   If a revert conflicts, run `git revert --abort`, stop, and show the user
   the conflicting paths; do not resolve the conflict without approval.
   Do not alter unrelated commits.

6. **Write `$GOOD` back and commit it.** The reverts restore the committed
   `progress.json`, whose `last_working_commit` names the checkpoint *before*
   `$GOOD`. Write `$GOOD` back so the next rollback starts from the right
   boundary:
   ```bash
   GOOD="$GOOD" python3 -c "import json, os
   path = '.nsight-ai-port/progress.json'
   p = json.load(open(path))
   p['last_working_commit'] = os.environ['GOOD']
   json.dump(p, open(path, 'w'), indent=2)
   open(path, 'a').write('\n')"
   git add .nsight-ai-port/progress.json
   git commit -m "nsight-port: roll back to $GOOD"
   ```

7. **Re-read** `progress.json` and the plan to understand
   where you are.

8. **Ask the user** what to do next — retry the phase, skip
   the function, or abort.
