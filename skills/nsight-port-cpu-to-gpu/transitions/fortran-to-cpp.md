---
from_lang: fortran
to_lang: cpp
status: complete
compiler: nvc++
cxx_standard: c++17
---
# Language Stage — Fortran -> C++ (CPU)

Transition: `transitions/fortran-to-cpp.md`. Stage A of a
cross-language port (`from_lang=fortran`, `to_lang=cpp`). This
file runs **before** the framework pack.

**Goal:** translate ONE Fortran function into a numerically-equivalent
**plain C++** function on the CPU, verified against the original Fortran
through an `iso_c_binding` / `extern "C"` interop test. No GPU code is
written here — the emitted C++ becomes the **"original"** that
`flavors/cpp/cuda/` or `flavors/cpp/kokkos/` then accelerates.

- Guard: `USE_CPP_TRANSLATION` — both the `-D` define and the build
  env var
- Output dir: sibling `{parent}/cpp/` next to the Fortran source

## Two-Stage Composition

```text
                          ┌─(USE_CUDA off)─> {fn}_cpp_host   (C++ CPU body)
caller.f90 ──(USE_CPP_TRANSLATION)──> {fn}_cpp
  Fortran                            dispatcher
                          └─(USE_CUDA on)──> {fn}_cuda       (Stage B)
```

- **Stage A (this guide):** Fortran `{fn}` -> C++, verified vs the
  Fortran reference. It emits **two** symbols:
  - `{fn}_cpp_host` — the translated CPU body. Contains **no**
    framework guard, ever, so it stays a genuine CPU implementation in
    every build configuration.
  - `{fn}_cpp` — a one-line dispatcher. In Stage A it just forwards to
    `{fn}_cpp_host`. This is the symbol the Fortran caller binds to,
    under `USE_CPP_TRANSLATION`.
- **Stage B (framework pack):** treat `{fn}_cpp_host` as the
  **original** and run the normal `flavors/{to_lang}/{framework}/`
  flow. It emits `{fn}_{framework}` (the framework suffix applied to
  the bare function name) and adds the framework guard to
  the **dispatcher only**, so the Fortran caller is unaware of the
  framework choice.

Stage A must fully pass (build + numeric verify + commit) before
Stage B starts.

## Directory Layout

Translated code goes in a sibling `cpp/` directory in the **same parent
folder** as the Fortran source — never collected into one project-wide
`cpp/` tree. For original `{parent}/{source}.f90` and target
`{function}`:

| Artifact | Path |
|----------|------|
| C++ CPU body + dispatcher | `{parent}/cpp/{function}_cpp.cpp` |
| Public decls (`extern "C"`) | `{parent}/cpp/{function}_cpp.hpp` |
| Interop test (**Fortran** program) | `{parent}/cpp/test/test_{function}/test_{function}.f90` |
| Shared comparison module | `$NSIGHT_AI_PORT_SOURCE_DIR/cpp/test/compare_field.f90` |

Every artifact above is per-function and sits beside the original it was
translated from, except the comparison module: it is identical for every
function, so the port keeps **one copy** at the source root rather than
one per source directory — see
[verification-guide.md](../references/verification-guide.md#one-copy-per-port-at-the-source-root).

The test is a Fortran program because Fortran is the original language;
see
[verification-guide.md](../references/verification-guide.md#test-main-must-be-in-the-original-language).
It needs no reference shim — it calls the original Fortran directly and
reaches the C++ candidate through the same `bind(C)` interface the
production call site uses.

Both symbols (`{function}_cpp_host` and `{function}_cpp`) live in that
one `.cpp` and are both declared in the `.hpp`. Stage B later edits only
the dispatcher in that file; the `_cpp_host` body is frozen once Stage A
passes.

The original Fortran file is never modified except the call site at
integration. Stage B reuses this same Fortran driver in-tree, pointing it
at the GPU symbol.

## Fortran -> C++ Translation Rules

- **Memory layout — keep it identical.** Fortran arrays are
  **column-major** and **1-based**; the C++ candidate operates on the
  **same flat buffer** passed by the driver, so index it the Fortran
  way. For `a(i,j)` with leading dimension `nx`:
  `a_cpp[(i-1) + nx*(j-1)]` (0-based flat). Do **not** transpose — a
  transpose shows up as large, structured diffs.
- **Bounds.** Pass every extent (and any non-1 lower bound) explicitly
  as `int`. Loop `do i=1,nx` becomes `for (int i=0;i<nx;++i)`; translate
  index expressions consistently.
- **Intent.** `intent(in)` -> `const T*` (or by-value scalar);
  `intent(out)`/`intent(inout)` -> `T*`. Scalars passed by value.
- **Types.** `real(8)`/`double precision` -> `double`; `real(4)` ->
  `float`; `integer` -> `int`; `logical` -> `int`/`bool` (be explicit
  about the Fortran `.true.` value when interoperating).
- **Modules / globals.** Module procedures -> free functions; module
  data the function reads must become explicit arguments (decouple in a
  Phase 2 refactor if needed). The candidate must be **flat-arg,
  externally callable** — no host-only types.
- **Interop signature.** Declare the candidate `extern "C"` so the
  Fortran caller can bind to it with a matching `bind(C)` interface.

### Intrinsics that do not mean the same thing in C

These are the silent ones: the code compiles, most cells agree, and a
handful of values at the edges of the domain are wrong. `modulo` in a
periodic-boundary helper is the classic — it corrupts exactly one cell per
array, which survives casual inspection and review.

| Fortran | Naive C | Correct C | Divergence |
|---|---|---|---|
| `modulo(a,p)` (floored) | `a % p` | `((a % p) + p) % p` | `modulo(-1,5)` is `4`; `-1 % 5` is `-1` |
| `mod(a,p)` (truncated) | `a % p` | `a % p` — correct as-is | Agrees; do not "fix" it into the floored form |
| `sign(a,b)` | `(b<0?-1:1)*fabs(a)` | `std::copysign(a,b)` | Differs at `b = -0.0`, where Fortran gives `-|a|` |
| `nint(x)` | `(int)(x+0.5)` | `std::lround(x)` | `nint(-2.4)` is `-2`; the naive form gives `-1` |
| `nint(x)` | `std::rint(x)` | `std::lround(x)` | `rint` is half-to-**even**: `rint(2.5)` is `2`, `nint(2.5)` is `3` |
| `a/b`, integers | `a/b` | `a/b` — correct as-is | Both truncate toward zero (`-7/2` is `-3` in each) |

Two traps worth stating plainly. First, Fortran has **both** `mod` and
`modulo` and they differ for negative arguments — check which one the
original used before reaching for `%`. Second, an index expression is the
worst place for this class of bug, because an off-by-`p` index reads
in-bounds garbage instead of crashing.

Prefer hoisting a periodic index into a small named helper with one
correct implementation, rather than repeating the `((i % n) + n) % n` form
at every use site.

## Stage A Steps

### A1: Build exploration (mixed-language link)

**Skip if** `.nsight-ai-port/build_notes.txt` already records the mixed
link strategy. Otherwise determine and record:

1. The toggle guard (`USE_CPP_TRANSLATION`).
2. How a `{parent}/cpp/` directory's `.cpp` files are compiled
   (`nvc++`/`g++`, `-std=c++17`) and linked into the Fortran program.
3. The **mixed-language link** approach. The test program is Fortran
   (see A2), so link with the Fortran driver and add the C++ runtime:
   `nvfortran ... -c++libs`. Record the exact invocation.
4. Name-mangling: the candidate is `extern "C"` and the Fortran side
   reaches it through a `bind(C)` interface, so there is no mangling
   guesswork in either direction.
5. **Preprocessing for every Fortran compiler that touches guarded
   source.** Once the guard lands in a shared `.f90`, *each*
   configuration that compiles that file needs preprocessing enabled —
   not just the one building the translated path. See
   [Preprocessing applies to every Fortran compiler](#preprocessing-applies-to-every-fortran-compiler)
   below and settle it now, not at A5.

### A2: Scaffold the interop test harness

**Subagent opportunity.** Inputs (signature + extents) and success
criterion (driver builds and runs against the original) are clear.

#### Prerequisite: externally callable

The Fortran target must be callable from a standalone program (a module
procedure with explicit arguments — not an internal/`contains`
subprogram, not reliant on host-scoped state). If not, the Phase 2
refactor was skipped — **STOP** and do it first. Do not substitute
whole-program verification.

#### Steps

1. **Read the Fortran function** — signature, argument intents, array
   shapes and leading dimensions, module dependencies.

2. **Create the test dir:**
   ```bash
   mkdir -p {parent}/cpp/test/test_{function}/
   ```

3. **No reference shim is needed.** The driver is a Fortran program (step
   6), so it `use`s the original module and calls `{function}` directly.
   That is the point of keeping the harness in the original language —
   see
   [Test `main` must be in the original language](../references/verification-guide.md#test-main-must-be-in-the-original-language).

4. **Write the C++ candidate header** `{function}_cpp.hpp` — declare
   **both** symbols:
   ```cpp
   #pragma once
   // CPU body: always a real CPU implementation, in every build.
   extern "C" void {function}_cpp_host(double* u, int n, double dt);
   // Dispatcher: what the Fortran caller binds to. Stage B guards this.
   extern "C" void {function}_cpp(double* u, int n, double dt);
   ```

5. **Write a placeholder candidate** `{function}_cpp.cpp` (real
   translation in A3):
   ```cpp
   #include "{function}_cpp.hpp"

   // ---- CPU body: NEVER add a framework guard (#ifdef USE_CUDA) here.
   extern "C" void {function}_cpp_host(double* u, int n, double dt) {
     // TODO(A3): translated body
   }

   // ---- Dispatcher: Stage B adds its USE_CUDA branch here, and only here.
   extern "C" void {function}_cpp(double* u, int n, double dt) {
     {function}_cpp_host(u, n, dt);
   }
   ```

6. **Write the Fortran interop driver** `test_{function}.f90` — a Fortran
   `program`, because Fortran is the original language. It calls the
   original directly and the C++ candidate through the **same `bind(C)`
   interface the real caller will use in A5**, so this test exercises the
   production interop path rather than a test-only wrapper:
   ```fortran
   program test_{function}
     use iso_c_binding
     use {original_module}, only: {function}        ! reference: called directly
     use nsight_port_compare                       ! copied in at the source root
     implicit none

     ! Same shape as the A5 call site: ADDRESS by value, no copy-in/out.
     interface
       subroutine {function}_cpp_host(u, n, dt) bind(C, name="{function}_cpp_host")
         use iso_c_binding
         type(c_ptr),    value :: u
         integer(c_int), value :: n
         real(c_double), value :: dt
       end subroutine
     end interface

     integer, parameter :: n = 1000        ! realistic UNIT size, not problem size
     real(c_double), parameter :: dt = 0.1_c_double
     ! Named for their role in the comparison — the harness takes
     ! (candidate, reference). TARGET so c_loc is valid.
     real(c_double), target :: output_reference(n), output_candidate(n)
     type(compare_tolerances) :: tol
     type(compare_result) :: res
     integer :: i

     ! Both paths start from identical values
     do i = 1, n
       output_reference(i) = 1.0_c_double + real(i - 1, c_double)
       output_candidate(i) = output_reference(i)
     end do

     call {function}(output_reference, n, dt)            ! reference (truth)
     call {function}_cpp_host(c_loc(output_candidate), n, dt)
                                                        ! CPU body, not the
                                                        ! dispatcher — so this
                                                        ! test is guard-independent

     tol = tolerances_for(output_reference(1))
     res = compare_field(output_candidate, output_reference, tol)
     if (.not. report_and_gate(res, tol, '{function}')) stop 1
     print *, 'TEST PASSED'
   end program
   ```
   The comparison harness ships with the skill as compilable source — do
   not re-derive it inline. Copy it **once for the whole port**, at the
   source root, so every Stage A and Stage B test in every directory
   reuses one implementation:

   ```bash
   mkdir -p "$NSIGHT_AI_PORT_SOURCE_DIR/cpp/test"
   cp assets/test-harness/compare_field.f90 \
      "$NSIGHT_AI_PORT_SOURCE_DIR/cpp/test/"
   ```

   The harness language follows `config.json:from_lang`, so this is the
   **Fortran** module even though the target language is C++: every test
   `main` is in the original language. Its contract:

   | Call | What it does |
   |------|--------------|
   | `tolerances_for(sample)` | kind-aware defaults, then `WMAD_TOL` / `ABS_TOL` / `REL_TOL` / `REL_FLOOR` / `MAX_REL_CHECK` from the environment |
   | `compare_field(candidate, reference, tol [, weight])` | generic over `real32`/`real64` at ranks 1–3 |
   | `report_and_gate(result, tol, label)` | prints `wmad`, `max_rel`, `n_exceed` of `n_total`, and the worst element's subscripts; returns `.true.` on pass |

   Rules: realistic **unit** sizes (never copy `problem` sizes);
   identical inputs to both paths; `wmad` is the gate and the outlier
   count is printed for review; reference files in-tree (no copying).

   A CPU-to-CPU translation should agree far more tightly than a GPU port
   does — same instruction set, same ordering unless you changed it. Treat
   any `n_exceed > 0` here as a translation bug (index mapping, precision,
   loop order) rather than numerics, and do not reach for
   `MAX_REL_CHECK=off` at this stage.

7. **Hook the build** — add a test target that compiles the original
   Fortran source and the Fortran driver (`nvfortran`), plus the C++
   candidate (`nvc++ -std=c++17`), then links them with the Fortran
   driver per A1 (`nvfortran ... -c++libs`). It must build under the
   normal build command. The comparison module is compiled **once for
   the whole port** into a single object that every test links; later
   functions add their driver to that object, not another copy of it.

8. **Write the manifest:**
   ```json
   {"test_executable": "/abs/path/to/test_{function}"}
   ```

9. **Verify the harness builds and the placeholder runs** (the
   placeholder will likely fail the numeric check — that is expected
   until A3; require only that it **builds and links**):
   ```bash
   source .nsight-ai-port/env.sh && bash .nsight-ai-port/build_correctness.sh
   ```
   Then git commit.

### A3: Translate the function to C++

Replace the placeholder body of **`{function}_cpp_host`**, following the
translation rules above. Work only in `{parent}/cpp/{function}_cpp.cpp`.
Keep the header plain (`extern "C"`, flat args). No GPU constructs, no
`#ifdef` — not even in the dispatcher yet.

The `{function}_cpp` dispatcher stays a forward to it, and carries a
one-time marker so A4 can prove the C++ path ran rather than the
original — see
[GPU Execution Marker](../references/verification-guide.md#gpu-execution-marker)
for the rank-0 form under MPI.

```cpp
extern "C" void {function}_cpp(double* u, int n) {
  static bool marker_printed = false;
  if (!marker_printed) {
    std::printf("NSIGHT_AI_PORT_LANG_EXEC: {function}\n");
    marker_printed = true;
  }
  {function}_cpp_host(u, n);
}
```

Verify after each logical chunk:
```bash
source .nsight-ai-port/env.sh && bash .nsight-ai-port/build_correctness.sh
./test_{function}
```
On failure, see
[Fault Tolerance](../SKILL.md#fault-tolerance). Common issues:
column-major/row-major mismatch (structured diffs), 1-based vs 0-based
off-by-one, `modulo` translated as `%` at a periodic boundary (see
[Intrinsics that do not mean the same thing in C](#intrinsics-that-do-not-mean-the-same-thing-in-c)),
`logical` truthiness, single/double precision mismatch, mixed-link
failures (wrong runtime libs). Git commit on success.

### A4: Verify numerical correctness

The driver already compares reference vs candidate. Confirm it prints
`TEST PASSED` with tolerances from `ABS_TOL`/`REL_TOL`:
```bash
source .nsight-ai-port/env.sh && ABS_TOL=1e-12 REL_TOL=1e-10 \
  bash .nsight-ai-port/run_correctness.sh   # if wired; else run ./test_{function}
```
Must pass before integration. Git commit.

### A5: Integrate (call-site dispatch + mixed build)

In the **Fortran caller** (not the candidate file), dispatch under the
guard via an `iso_c_binding` interface. **Pass array data as an address
by value (`type(c_ptr), value` + `c_loc`), NOT as an array dummy** — see
the gotcha below.

```fortran
#ifdef USE_CPP_TRANSLATION
  use iso_c_binding, only: c_ptr, c_loc, c_int, c_double
  interface
    subroutine {function}_cpp(u, n, dt) bind(C, name="{function}_cpp")
      use iso_c_binding
      type(c_ptr),    value :: u      ! ADDRESS by value (no copy-in/out)
      integer(c_int), value :: n
      real(c_double), value :: dt
    end subroutine
  end interface
#endif
  ! The actual array must have the TARGET attribute and be contiguous so
  ! c_loc yields a stable address:   real(8), target :: u(:)
  ! ...
#ifdef USE_CPP_TRANSLATION
  call {function}_cpp(c_loc(u), n, dt)             ! pass &u(1), not u
#else
  call {function}(u, n, dt)
#endif
```

The C++ candidate keeps its plain `double* u` signature — a `c_ptr`
passed by value matches a C pointer argument in the ABI.

#### Gotcha: nvfortran copies array args to `bind(C)` dummies

If the interface declares an **array dummy**
(`real(c_double) :: u(*)` / `u(:)`), nvfortran makes a **contiguous
host temporary** of the actual argument (copy-in on call, copy-out on
return). For a plain CPU language stage this is merely wasteful, but it
**silently defeats resident managed memory in Stage B**: the C++/CUDA
side receives the address of the *temporary*, not the managed buffer, so
kernels run on transient host memory and `cudaMallocManaged`
residency/coherence is lost (no error, just wrong/slow behavior).

**Fix (do this from the start so Stage B inherits it):** declare the
pointer as `type(c_ptr), value` and pass `c_loc(array)`. Give every
array actual argument the `target` (or `pointer`) attribute. This
forwards the real device-resident address with no hidden copy. The same
rule applies to the A2 test driver's own `bind(C)` interface: the CPU-only
correctness test tolerates array dummies, but use the `c_loc` form there
too, so the test exercises the same calling convention as production.

#### Preprocessing applies to every Fortran compiler

Adding `#ifdef` dispatch to a shared `.f90` changes the requirements for
**every configuration that compiles that file**, not just the one building
the translated path. The moment the guard lands, any Fortran compile line
lacking preprocessing stops building the file at all.

Enumerate the compilers that touch the guarded source and give each one
its preprocessing flag plus the define it needs:

| Compiler | Enable preprocessing | Notes |
|---|---|---|
| `nvfortran` | `-Mpreprocess` | Or rename the file to `.F90`. |
| `gfortran` | `-cpp` (`-fpp` on some builds) | Needed for lowercase `.f90`. |
| `ifx` / `ifort` | `-fpp` | |

The case that is easy to miss: a project that keeps a **pure-Fortran CPU
baseline built by `gfortran`** while the translated path uses `nvfortran`.
That baseline compiles the same guarded file, so it needs `-cpp` even
though it never enables the guard. Symptom is not a wrong answer but a
build that breaks the instant dispatch is added.

Settle this during A1 rather than at A5 — a project with more than one
Fortran configuration will hit it, so treat it as expected work. Prefer
adding the flag over renaming sources to `.F90` project-wide. Then verify
by building **every** configuration, not only the translated one.

Build-system updates:
- Compile every `{parent}/cpp/*.cpp` with `nvc++ -std=c++17` and add a
  `-DUSE_CPP_TRANSLATION` define to the guarded Fortran sources when
  `USE_CPP_TRANSLATION=1`.
- **Preprocessor defines must reach the Fortran compiler too.** Any
  `#ifdef` symbol used in `.f90`/`.F90` source (here
  `USE_CPP_TRANSLATION`, and in Stage B `USE_CUDA`) must be passed to the
  Fortran compiler with preprocessing enabled, **and** to *every* Fortran
  compiler that touches the guarded file — see
  [Preprocessing applies to every Fortran compiler](#preprocessing-applies-to-every-fortran-compiler).
- Apply the mixed-link strategy from A1 so the Fortran program links the
  C++ objects (and C++ runtime).
- Update parent archive/link rules so `{parent}/cpp/*.o` is included
  (globs over `*.f90` will not match the new subdir).
- **Never** pass these flags to external/third-party libraries.
- **Separate object trees per configuration**
  (`$NSIGHT_AI_PORT_BUILD_DIR/cpu` and `/gpu`), so alternating CPU and
  GPU builds stays incremental and correct. Never clean
  unconditionally — see
  [never clean unconditionally](../references/verification-guide.md#never-clean-unconditionally).

Dual-path verification (the CPU language gate):
```bash
# 1. Original Fortran path
source .nsight-ai-port/env.sh && bash .nsight-ai-port/build_correctness.sh
source .nsight-ai-port/env.sh && bash .nsight-ai-port/run_correctness.sh
# 2. C++ language path + marker
source .nsight-ai-port/env.sh && \
  USE_CPP_TRANSLATION=1 bash .nsight-ai-port/build_correctness.sh
source .nsight-ai-port/env.sh && \
  USE_CPP_TRANSLATION=1 bash .nsight-ai-port/run_correctness.sh
```
Step 2 stdout MUST contain `NSIGHT_AI_PORT_LANG_EXEC: {function}`;
missing marker is a FAILURE even at exit 0 (the C++ path did not run —
usually `-DUSE_CPP_TRANSLATION` not reaching the caller or the C++
objects not linked). This run must exercise the production dispatcher with
`USE_CPP_TRANSLATION=1` and the framework guard disabled and omitted from
the preprocessor defines. Its output MUST NOT contain
`NSIGHT_AI_PORT_FRAMEWORK_EXEC: {function}`; that marker means the supposed
CPU path ran the framework implementation. The source must also keep the
two-symbol invariant: `{function}_cpp_host` contains no framework guard, and
the CPU comparison reference binds directly to `{function}_cpp_host`, never
`{function}_cpp`. A missing language marker or a present framework marker
blocks the Stage A commit and Stage B.

Git commit, then update `progress.json`: set this function's
`phases.language_translate: true` and update `last_working_commit`.

## Handoff to Stage B (framework acceleration)

Framework-pack `translation.md` files **do not repeat** these rules.
When `from_lang != to_lang`, Stage B reads this section and follows
it.

**Stage B must:**

1. Treat `{function}_cpp_host` as the CPU reference — never the
   `{function}_cpp` dispatcher.
2. Reuse Stage A's Fortran driver in-tree; do not write a C++ `main`.
3. Reuse Stage A's `compare_field.f90`; do not copy
   `compare_field.hpp`.
4. Gate GPU and CPU-host against the **original Fortran**. Report
   CPU-vs-GPU; do not gate on it.
5. Add a `bind(C)` interface for `{function}_{framework}` on that
   same driver.
6. Emit GPU files under `{parent}/{framework}/`, not under
   Stage A's `cpp/` tree.
7. Repeat the Stage A guard/reference gate after every dispatcher, guard, or
   build-system change: translation enabled plus framework disabled must emit
   `NSIGHT_AI_PORT_LANG_EXEC: {function}` and no
   `NSIGHT_AI_PORT_FRAMEWORK_EXEC: {function}`; the three-way driver must call
   the host and framework symbols separately and must not use the dispatcher
   as its CPU reference.

The verified **`{function}_cpp_host`** is now the **original** for the
framework pack. Run `flavors/{to_lang}/{framework}/` (e.g.
`flavors/cpp/cuda/`) with that symbol as the source. It emits
`{parent}/cuda/{function}_cuda.{cu,hpp}` — the framework's own
directory and suffix, not nested under the Stage A `cpp/` tree — and
adds the `USE_CUDA` branch to the **dispatcher only**:

```cpp
extern "C" void {function}_cpp(double* u, int n, double dt) {
#ifdef USE_CUDA
  {function}_cuda(u, n, dt);
#else
  {function}_cpp_host(u, n, dt);
#endif
}
```

`{function}_cpp_host` is untouched by Stage B. The Fortran caller's
`USE_CPP_TRANSLATION` dispatch is unchanged. Both guards must be enabled
together for the full GPU build (`USE_CPP_TRANSLATION=1 USE_CUDA=1`).

Naming: apply `_{framework}` to the **bare function name**, so the
GPU symbol is **`{function}_cuda`**. Stage A's suffixes do not accumulate
onto it — never `{function}_cpp_cuda` or `{function}_cpp_host_cuda`. The
trio reads

| Symbol | Role |
|--------|------|
| `{function}_cpp` | dispatcher (what Fortran binds to) |
| `{function}_cpp_host` | CPU C++ body |
| `{function}_cuda` | GPU implementation |

For a Kokkos target substitute `_kokkos`; the pattern is
`{function}_{framework}`.

### The reference for every comparison is the original Fortran

The examples below use Fortran as the original language. The rule itself
is language-neutral — gate against the original code in whatever
language it is written in — see
[The reference is always the original language](../references/verification-guide.md#the-reference-is-always-the-original-language).

Stage B's test has three implementations available, and should exercise
all three on identical inputs:

| Symbol | Role |
|--------|------|
| `{function}` (called directly by the Fortran driver) | **Source of truth** |
| `{function}_cpp_host` | Stage A CPU translation |
| `{function}_{framework}` | Stage B GPU translation |

The pass/fail gate is **agreement with the Fortran**, never
CPU-versus-GPU:

- **GPU vs Fortran** — the round's regression signal. Must pass.
- **CPU-host vs Fortran** — must keep passing. Catches a Stage A
  regression or build damage masquerading as a GPU bug.
- **CPU-host vs GPU** — report it, but do **not** gate on it. A
  disagreement between the two new implementations only matters because
  it means at least one of them disagrees with the Fortran, and the two
  checks above already say which.

Reuse Stage A's Fortran driver in-tree; do not copy it. It stays a Fortran
program in Stage B, adding a `bind(C)` interface to
`{function}_{framework}` alongside the one it already has for
`{function}_cpp_host`.

Use the triangulation to localize a failure:

| GPU vs Fortran | CPU-host vs Fortran | Read it as |
|----------------|---------------------|------------|
| fail | pass | A Stage B (GPU) defect — kernel, indexing, or memory mode. |
| fail | fail | Stage A or shared build/config is broken. Fix Stage A first; do not debug kernels. |
| pass | fail | Stage A regressed after handoff (or the host path is no longer built). |
| pass | pass | Round is good, even if CPU and GPU differ slightly from each other. |

Both A5 gotchas carry into Stage B and bite managed mode: keep the
[`c_loc` call site](#gotcha-nvfortran-copies-array-args-to-bindc-dummies),
and pass the framework guard (e.g. `-DUSE_CUDA`) to
[every Fortran compiler](#preprocessing-applies-to-every-fortran-compiler),
not only nvcc/C++.
