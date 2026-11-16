# Setup Guide (Phase 0)

## Start-of-Run Reconciliation (do this first)

Before the interview, check whether this project has already been set
up. Read `{source_dir}/.nsight-ai-port/progress.json` if it exists:

| State | What it means | Action |
|-------|---------------|--------|
| No `.nsight-ai-port/` | Brand-new project | **Fresh setup** — full interview below |
| `status: "in_progress"` | A previous run was interrupted | **Resume** — do **not** re-interview; continue the nsys round loop |
| `status: "completed"` | A prior port finished | **Reconcile** — apply the flow below |

**Fresh setup** and **resume** are the existing flows. The third case —
a *completed* project that the user invokes again (e.g. to port a new
problem) — is the composable path. Handle it explicitly so a new run
never clobbers or is confused by the finished one.

### Completed project → reconcile

When `status: "completed"`, offer three choices. Under
`config.json:engagement == "low"`, default to **(a)** without asking;
otherwise ask the user:

- **(a) Add problem(s)** *(default)* — keep everything: git history,
  the cumulative `translated_functions`, the correctness scripts, and
  the flavor config. **Archive** the finished run (see
  "Archiving Runs"), then run the **scoped mini-interview** (only Q3
  and Q4, for the new workload[s]) and append the new problem(s) to
  `config.json:problems[]` and `progress.json:problems[]` with
  `status: "pending"`. Set `progress.json:status` back to
  `"in_progress"`. The round loop then runs only the not-yet-completed
  problems.
- **(b) Start over** — archive the entire finished run, reset
  `progress.json` from the template, and run the **full interview**.
- **(c) Inspect only** — summarize the finished port from
  `progress.json` / `history/` and stop.

The scoped mini-interview reuses the existing `from_lang` /
`to_lang` / `framework`, `memory_mode`, `translation_mode`,
`include_guard`, and correctness scripts. Only the performance
workload(s) are re-asked. Do **not** re-run the full interview for
**(a)**. 

## GPU Environment Check

> **When:** *after* the [User Interview](#user-interview) and after
> `config.json` exists — the flavor id comes from Q2 and the results are
> written into the config. Do not run this while reading top-to-bottom;
> the invocation point is
> [Creating `.nsight-ai-port/`](#creating-nsight-ai-port). This section
> documents what the check covers.

Run `scripts/gpu-check.sh` with the flavor id from the interview
(e.g. `fortran-openacc`):

```bash
scripts/gpu-check.sh fortran-openacc
```

The script checks `nvidia-smi`, git, and the flavor compiler from
the nested pack `flavors/{lang}/{framework}/manifest.json` (looked
up by `manifest.id`, e.g. `fortran-openacc`). If anything is missing
or the GPU is not accessible, report the failure and stop.

Record in `.nsight-ai-port/config.json:gpu_environment`:
- `compiler_path`: output of `which <compiler>` from manifest
- `compute_capability`: from `nvidia-smi --query-gpu=compute_cap`
- `cuda_version`: from `nvidia-smi` driver output

## Ensure Source Dir Is Its Own Git Repo

> **When:** as soon as Q1 supplies the source directory, before any file
> is written into it.

After the user provides the source directory, check whether it
is already a git repository:

```bash
cd "$SOURCE_DIR"
git rev-parse --git-dir 2>/dev/null
```

Three cases:

1. **Not a git repo at all.** Create one with the
   [reviewed initial commit](#reviewed-initial-commit).

2. **Git root is a parent directory** (the source dir is nested
   inside a larger repo). This is dangerous — porting commits
   would pollute the parent project, and deleting the source dir
   later leaves orphaned commits. Create an isolated repo in the
   source dir with the
   [reviewed initial commit](#reviewed-initial-commit), and tell the
   user the parent repo will now see the source dir as a nested repo.

3. **Already its own git repo** (git root == source dir). Good,
   no action needed.

To detect case 2, compare the git root to the source dir:
```bash
GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
[ -n "$GIT_ROOT" ] && [ "$GIT_ROOT" != "$SOURCE_DIR" ] && echo nested
```

### Reviewed initial commit

Initializing a repo and committing everything is the user's decision,
at every engagement level:

1. Tell the user which case applies and that you propose to run
   `git init` in `$SOURCE_DIR`. Proceed only after explicit approval;
   if they decline, stop setup and ask how they want to version the
   source.
2. Initialize and stage, then show what would be committed:
   ```bash
   git init
   git add -A
   git status --short
   ```
3. Review the staged list with the user. Flag build outputs, large
   binaries or data files, and anything that looks like a credential
   (`.env`, keys, tokens). For each path the user excludes, unstage it
   and add it to `.gitignore`:
   ```bash
   git rm -r --cached --quiet -- path/to/excluded-file
   ```
4. Commit only after the user approves the final staged list:
   ```bash
   git commit -m "initial commit"
   ```

## User Interview

Ask these questions in order. Skip any the user has already answered in
their initial message. Do NOT analyze source code or propose specific
porting targets during the interview — that happens in Phase 1.

**The information each question elicits is required; the wording is a
default, not a script.** Prefer the phrasing below — a consistent
interview is easier for users who run the skill more than once — but
adapt it where the conversation warrants: fold two questions together
when one answer settles both, drop options that cannot apply to this
project, and match the user's own vocabulary for their codebase. What you
may not do is skip a question or infer its answer silently; if you
reword, the same decision must still be put to the user.

Every answer maps to a named field in
`.nsight-ai-port/config.json`. The template is in
`assets/config-template.json`.

### Q1: Source directory → `source_dir`

> "Where is the source code I should work with?"

Must be an absolute path. It is also the anchor for the one shared
comparison harness the port copies in — `{source_dir}/{framework}/test/`
(cross-language: `{source_dir}/{to_lang}/test/`), see
[verification-guide.md](verification-guide.md#one-copy-per-port-at-the-source-root)
— so answer with the root of the source tree being ported rather than a
subdirectory inside it.

### Q1b: Detect source language → `from_lang`

Do NOT ask this — **auto-detect** it from the source tree, then
confirm with the user. Count source files by extension under
`source_dir`:

```bash
# Fortran: .f90 .F90 .f .F .f95 .F95 ;  C/C++: .cpp .cc .cxx .C .c++ .c
```

Set `from_lang` to the dominant language (`fortran` or `cpp`) and
confirm: "I detected this is a **{from_lang}** codebase — correct?"
If the user overrides, use their value.

### Q2: GPU porting target → `to_lang`, `framework`

Offer only the targets whose `to_lang` equals `from_lang`, or for
which `transitions/{from_lang}-to-{to_lang}.md` exists with
frontmatter `status: complete`. This is the only supported-pair
check; later stages assume it passed. (Today: a Fortran codebase
sees all four; a C++ codebase sees only (b) and (c).)

> "What GPU target should I port to? Pick one:"
>
> **a) Fortran + OpenACC** — nvfortran `-acc`; directives in
> sibling `openacc/` directories.
> **b) C/C++ + Kokkos** — Kokkos 4 views/functors in sibling
> `kokkos/` directories; `nvcc_wrapper`/`nvc++`, C++17.
> **c) C/C++ + CUDA** — CUDA `.cu` kernels in sibling `cuda/`
> directories; `nvcc`, C++17.
> **d) Fortran + OpenMP offload** — nvfortran `-mp=gpu`;
> `!$omp target` directives in sibling `openmp/` directories.

| Choice | `to_lang` | `framework` |
|--------|-----------|-------------|
| (a) | `fortran` | `openacc` |
| (b) | `cpp` | `kokkos` |
| (c) | `cpp` | `cuda` |
| (d) | `fortran` | `openmp` |

`from_lang` comes from Q1b; `to_lang` + `framework` come from the
choice above. Write all three to `config.json`.

After the user chooses, read
`flavors/{to_lang}/{framework}/manifest.json`. All four framework
targets are `"status": "complete"` today. If a flavor is marked
`"status": "stub"`, stop and tell the user it is not yet fully
supported — do not proceed.

Set `include_guard` to `USE_{FRAMEWORK}` (upper-cased `framework`)
unless overridden in Q7.

#### Cross-language porting (`from_lang != to_lang`)

If the detected `from_lang` differs from the chosen `to_lang`, the
port is **two-stage** per function:

1. **Language stage** —
   `transitions/{from_lang}-to-{to_lang}.md` translates the
   function to `{to_lang}` on the CPU and verifies it against
   the original via interop.
2. **Framework stage** — `flavors/{to_lang}/{framework}/`
   accelerates the emitted `{to_lang}`.

Tell the user: "Your source is **{from_lang}** but the target is
**{to_lang}** — I'll translate each function to {to_lang} first
(verified against the original), then accelerate it with
{framework}." The router SKILL.md "Cross-Language Porting"
section drives the two-stage flow; no extra config field is
needed (it is inferred from `from_lang != to_lang`).

### Q3: Performance workload(s) → `performance_mode`, `problems[]`

The performance workload(s) drive the port: nsys profiles each
one. A workload only exercises — and can therefore only get ported —
the compute code it actually **runs**. Large codes rarely hit their
full compute path with one problem, so first decide the **coverage
mode**, then capture the workload(s).

**Step 1 — Coverage mode → `performance_mode`:**

> "Do you want to port against a **single** performance workload, or a
> **sequence** of workloads that each exercise a different compute path
> (so we can build up to a fuller port)?"

| Choice | `performance_mode` |
|--------|--------------------|
| Single workload | `"single"` (default) |
| Sequence of workloads | `"sequence"` |

**Step 2 — Capture the workload(s):**

Each workload is a **problem** object with a short stable `id`, a
`description`, and (in Q4) its own build/run scripts.

- **single** — capture one representative case:
  > "What **single workload** should I use to measure performance? One
  > case — e.g. one benchmark, one input deck, N timesteps — **not**
  > your full test suite."

  Write one entry with id `"main"` to `config.json:problems[]`.

- **sequence** — capture (or derive) an **ordered set** of cases, one
  per distinct compute path:
  > "List the workloads (one per compute path), or describe the code
  > and I'll propose a covering set."

  Write each as an entry in `config.json:problems[]` (`id`,
  `description`, `notes`). Order matters — front-load shared/core
  compute so later cases reuse earlier ports.

**Generating workloads for the user.** If the user asks you to author
the workload(s) rather than supplying them, read
[coverage-set-guide.md](coverage-set-guide.md): it explains how to
build a "typical use" single workload, or enumerate compute-path axes
(mining the project's own regression/CI matrix and example decks) into
a minimal ordered coverage set. Present the proposed set for approval.

Rules:

- Each problem focuses on **one** compute codepath — not "run all
  regression tests."
- `performance_mode: "single"` ⇒ exactly one entry in `problems[]`.

### Q4: Performance scripts → per-problem `build_performance` / `run_performance`

Ask immediately after Q3. **Each** problem from Q3 gets its **own**
build + run pair under `.nsight-ai-port/`, separate from the correctness
pair (Q6). They are separate because the build may differ — e.g. the
grid size is baked in at compile time, so the perf build compiles a
different case than the correctness build.

> "For each workload, how do I build and run it? Point me at existing
> scripts, give me a one-shot command, or I'll help draft them."

For every problem in `problems[]`:

- **Existing scripts** — copy into `.nsight-ai-port/` (same copy-in rule
  as Q6, never edit the originals):
  ```bash
  cp "<user_perf_build>" .nsight-ai-port/build_performance_{id}.sh
  cp "<user_perf_run>"   .nsight-ai-port/run_performance_{id}.sh
  chmod +x .nsight-ai-port/build_performance_{id}.sh \
    .nsight-ai-port/run_performance_{id}.sh
  ```
- **One-shot command / draft** — author
  `.nsight-ai-port/run_performance_{id}.sh` (exactly one program
  invocation; see "Performance Scripts" below) and
  `.nsight-ai-port/build_performance_{id}.sh`. If a perf build is
  identical to correctness, it may simply mirror `build_correctness.sh`.
  To author a covering **set** of workloads, follow
  [coverage-set-guide.md](coverage-set-guide.md). Get user approval.

**Script naming.** A lone `single`-mode case may use the plain
`build_performance.sh` / `run_performance.sh` (id `"main"`). Multiple
problems use `_{id}`-suffixed names so they never clobber each other.
The paths recorded in `config.json` are authoritative regardless of
name.

Before assigning ids, read the existing `config.json:problems[]`.
Every new id and script path must be unique. Never rename, replace, or
overwrite an existing problem or performance script. When numeric ids
are used, continue after the highest existing number.

Set each `problems[].build_performance` / `problems[].run_performance`
to its `.nsight-ai-port/` paths and fill its metadata. At runtime,
`progress.json:current_problem` identifies the active entry by id;
resolve all problem metadata and scripts from that `problems[]` entry.

```json
"problems": [
  {
    "id": "main",
    "description": "From Q3",
    "build_performance": ".nsight-ai-port/build_performance.sh",
    "run_performance": ".nsight-ai-port/run_performance.sh",
    "working_dir": "",
    "notes": "Inputs, env vars, expected runtime",
    "expected_exit_code": 0,
    "covers": "Which compute path this workload exercises"
  }
]
```

| Field | Meaning |
|-------|---------|
| `run_performance` | The perf entrypoint (one invocation). nsys profiles it. |
| `working_dir` | Optional; empty = run from `source_dir` after `cd` in script |
| `expected_exit_code` | Default `0`; a non-matching exit is a failure |
| `covers` | Optional; the compute path this problem exercises (sequence mode) |

Each `run_performance` must run **exactly one** case — never the full
test suite. The correctness suite (Q6) is for numerical checks only.

#### Step 3 — time the workload before accepting it

Do not accept a performance workload you have not timed. A shipped example
case is often a smoke test that finishes in milliseconds, which is
**unprofileable**: nsys collects no CPU samples, there is nothing to
attribute hotspots to, and no GPU could help it regardless.

Run each `run_performance` script once and record the wall time:

```bash
/usr/bin/time -f '%e s' bash .nsight-ai-port/run_performance_{id}.sh
```

| Wall time | Action |
|---|---|
| **< ~1 s** | Too short. Do not accept it — propose a scaled variant and get approval. |
| ~1–10 s | Good target for the porting loop. |
| > ~10 min | Ask whether a smaller case still exercises the same path. |

When scaling, prefer **more work per run** in this order: larger problem
size, more timesteps, then repeating the compute loop. Record the original
and scaled settings in the problem's `notes`.

**Make the size a runtime setting, not a compile-time constant.** If the
grid size is a `parameter` baked in at compile time, the perf and
correctness configurations need two different *builds*, and every later
round pays that cost. Promoting it to a command-line argument or
environment variable lets one build serve both — a small refactor at setup
that removes a recurring tax. Propose it when you find a compile-time
size.

Record the measured wall time in `notes` either way. Round 0
re-checks profileability (see
[profiling-guide.md](../orchestrators/nsys/references/profiling-guide.md));
timing it here means that check confirms a decision already made rather
than forcing a restart after setup.

### Q5: Engagement level → `engagement`

> "How involved do you want to be in reviewing changes?"
>
> **a) Low** (default) — Between approval points I'll work
> autonomously and report at the end of each round, rather than
> showing you every diff. Best when you trust the process and
> want speed.
>
> **b) High** — After every major change I'll also show you a
> diff with an explanation, and wait for you to accept, reject
> with comments, or ask questions before I proceed. Best for
> unfamiliar codebases or when you want to learn as we go.
>
> Either way I'll still stop for the fixed approval points: the
> plan, the verify scripts, the workload, and whether to run
> another round.

| Choice | `engagement` |
|--------|-------------|
| (a) | `"low"` |
| (b) | `"high"` |

### Q6: Correctness scripts → `build_correctness`, `run_correctness`

> "Do you have existing build and run scripts for verifying correctness?"

Verification scripts are **always owned by `.nsight-ai-port/`**. The skill
never reads or edits build/run scripts that live elsewhere in (or
outside) the source tree — it works only on copies under
`.nsight-ai-port/`.

- **Yes:** **Copy** each script the user points to into
  `.nsight-ai-port/` before using it, then record the **copy** paths:
  ```bash
  cp "<user_build_script>" .nsight-ai-port/build_correctness.sh
  cp "<user_run_script>"   .nsight-ai-port/run_correctness.sh
  chmod +x .nsight-ai-port/build_correctness.sh .nsight-ai-port/run_correctness.sh
  ```
  Set `build_correctness` / `run_correctness` to
  `.nsight-ai-port/build_correctness.sh` / `.nsight-ai-port/run_correctness.sh`
  — never the originals. All later edits (GPU flags at integration, etc.)
  target the copies only; the user's originals stay untouched. If a
  copied script used paths relative to its old location, fix the **copy**
  so it `cd`s to `$NSIGHT_AI_PORT_SOURCE_DIR` (see env.sh usage). Verify
  each copy is executable and exits 0 on success / non-zero on failure.
- **No:** Author new scripts directly under `.nsight-ai-port/` as
  `.nsight-ai-port/build_correctness.sh` and
  `.nsight-ai-port/run_correctness.sh` (see "Proposing Verification
  Scripts" below).

**`run_correctness` is the numerical correctness gate** — it may run
many tests or cases. It is **not** the performance/profiling target;
that is `run_performance` (Q4).

### Q7: Include guard → `include_guard`

> "Does this project use a custom preprocessor guard for GPU
> code? If unsure, I'll detect it during build system analysis."

Default: `USE_{FRAMEWORK}` (e.g. `USE_OPENACC`, `USE_CUDA`).

### Q8: Memory mode → `memory_mode`

Ask only when the flavor manifest has `"supports_memory_mode": true`
(fortran-openacc, fortran-openmp, cpp-kokkos, and cpp-cuda today).
Otherwise set
`"separate"` and skip the question.

> "How should GPU memory be managed?"
>
> 1. **Managed** (recommended) — Software-level memory management
>    (UVM). The allocations themselves change to a managed
>    allocator, and the CUDA runtime migrates pages on demand; you
>    add no explicit copies. Runs on **any** CUDA GPU, discrete
>    included. OpenACC: `-gpu=mem:managed`. OpenMP:
>    `-gpu=mem:managed` + `!$omp requires
>    unified_shared_memory`. Kokkos: `Kokkos::SharedSpace`
>    Views (Kokkos 4). CUDA: `cudaMallocManaged` (no
>    `cudaMemcpy`). Requires rewriting allocation sites — in a
>    mixed-language build that means the **caller's** allocations
>    too (e.g. a Fortran `cudaMallocManaged` + `c_f_pointer`
>    shim), not just the translated kernels.
> 2. **Explicit** — Discrete GPU memory. You control
>    data movement explicitly, hoisted to the widest stable
>    scope (never per-kernel on hot paths). OpenACC: `!$acc
>    data` / `copyin` / `copyout`. OpenMP: `!$omp target data`
>    / `map(to:)` / `map(from:)`. Kokkos: device Views + host
>    mirror + `deep_copy`. CUDA: `cudaMalloc` + `cudaMemcpy`.
> 3. **Unified** — Hardware-coherent memory (e.g. GH200, or HMM
>    on supported platforms). Ordinary host allocations are
>    already device-accessible, so **no** allocator changes and
>    no explicit copies. Correct only where the hardware
>    provides coherence — on a discrete GPU without it, choose
>    `managed` instead. OpenACC: `-gpu=mem:unified`. OpenMP:
>    `-gpu=mem:unified` + `!$omp requires
>    unified_shared_memory`. Kokkos: non-owning unmanaged Views
>    over host pointers. CUDA: pass host pointers directly to
>    kernels.

The three modes differ in **who provides coherence**: you
(`separate`), the CUDA runtime through a managed allocator
(`managed`), or the hardware (`unified`). Only `unified` depends
on the platform; `managed` is portable but obliges you to change
every allocation that a kernel touches.

Map the choice to `memory_mode`:
- 1 → `"managed"`
- 2 → `"separate"`
- 3 → `"unified"`

Default: `"managed"`.

**Recommend `managed` unless the user says otherwise.** It removes the
largest source of hand-written, easy-to-get-wrong code in a port —
every copy direction, every residency lifetime, every stale-data bug —
and it is the only mode that is both portable and copy-free. Hoisting
explicit transfers to the widest stable scope is real work that has to
be redone whenever the call graph changes, and getting it subtly wrong
produces wrong answers rather than a build error.

A **discrete GPU is not a reason to avoid `managed`.** On-demand page
migration works on every CUDA GPU since Pascal, so "the GPU is discrete"
argues for `managed` over `unified`, never for `separate`. Do not
present `separate` as the safer choice for discrete hardware.

Choose `separate` when there is a concrete reason, not by default:

- Profiling shows page-migration cost dominating — heavy HtoD/DtoH
  traffic or high migration counts for data that could simply stay
  resident. This is a **Round N** finding, not a setup-time guess.
- The project already has hoisted data regions or explicit device
  buffers, so matching the existing style is less churn than converting
  the allocation sites.
- The allocation sites cannot be changed (third-party or generated
  allocator, or a caller you do not own).

`memory_mode` is recorded in `config.json` and is not a one-way door: if
a later profiling round shows migration overhead, switch the mode and
re-run the affected phases. Starting from `managed` and specializing
after measurement is cheaper than hand-writing transfers up front and
discovering they were unnecessary.

Record the choice in `config.json` and use it throughout
Phases 4c–4e (subskill translate-function). The flavor pack's
`patterns.md` defines the exact per-mode mechanics. See "Memory
mode → compiler flags" below.

### Q9: Translation placement → `translation_mode`

Ask only when the flavor manifest has
`"supports_inline_translation": true` (fortran-openacc and
fortran-openmp today — directive flavors whose `!$acc`/`!$omp`
lines are inert comments without the GPU flag). Otherwise set
`"sibling"` and skip the question.

> "Where should the GPU version of each function live?"
>
> **a) Sibling files** (default) — Translated GPU code goes in new
> `{parent}/{framework}/` files; your original sources stay
> untouched except for `#ifdef` call-site dispatch at integration.
> Safest blast radius; the original and GPU functions coexist.
>
> **b) Inline** — The original function is converted in place to
> the GPU version (directives added directly). A frozen, renamed
> copy of the original is saved as the CPU reference for the
> standalone test. Produces cleaner, integrated code with no
> parallel `_{framework}` functions, but edits your real
> sources directly.

| Choice | `translation_mode` |
|--------|--------------------|
| (a) | `"sibling"` |
| (b) | `"inline"` |

Default: `"sibling"`. Inline mode is only valid for the two
Fortran directive flavors; for C++ CUDA/Kokkos always use
`"sibling"`. The translate-function subskill opens the flavor pack's
`translation.md` and follows its routing based on `translation_mode`;
that router then redirects to the mode-specific guide
(`translation-sibling.md` or `translation-inline.md`). Do not point
readers at the mode-specific files directly.

### Memory mode → compiler flags

Map `config.json:memory_mode` to the flavor's build flags and
record the mapping in `.nsight-ai-port/build_notes.txt`.

**Fortran + OpenACC (nvfortran):**

| `memory_mode` | nvfortran memory flag | Data directives in GPU code |
|---------------|----------------------|----------------------------|
| `separate` | (none — `-gpu=ccNN` only) | Hoisted `!$acc data` / `enter data`; never per hot-call kernel |
| `managed` | `-gpu=mem:managed` | **None** — compute directives only |
| `unified` | `-gpu=mem:unified` | **None** — compute directives only |

Example GPU flag line for unified mode on cc 9.0:
```bash
-acc -gpu=cc90,mem:unified -DUSE_OPENACC
```

Wire this into the build scripts when `USE_OPENACC=1`. The memory
flag must match `config.json:memory_mode` — do not set
`managed` in config and `mem:unified` in the build script.

**Fortran + OpenMP offload (nvfortran):**

| `memory_mode` | nvfortran flags | Directives in GPU code |
|---------------|-----------------|------------------------|
| `separate` | `-mp=gpu -gpu=ccNN` | Hoisted `!$omp target data` / `enter data`; never per hot-call kernel |
| `managed` | `-mp=gpu -gpu=ccNN,mem:managed` | **None** — compute only + `!$omp requires unified_shared_memory` |
| `unified` | `-mp=gpu -gpu=ccNN,mem:unified` | **None** — compute only + `!$omp requires unified_shared_memory` |

Example GPU flag line for unified mode on cc 9.0:
```bash
-mp=gpu -gpu=cc90,mem:unified -DUSE_OPENMP
```

**Use `-mp=gpu`, never `-fopenmp`** — `-fopenmp` on nvfortran
enables host multithreading only, not device offload. Note the
OpenMP default: arrays in a `target` region are implicitly mapped
`tofrom`, so `separate` mode silently copies unless data regions
are hoisted, and unified/managed modes need `!$omp requires
unified_shared_memory` to suppress those implicit maps. See
`flavors/fortran/openmp/build-integration.md`.

**C/C++ + Kokkos (nvcc_wrapper / nvc++, Kokkos 4 + C++17):**

| `memory_mode` | Kokkos build requirement | View strategy in GPU code |
|---------------|--------------------------|---------------------------|
| `separate` | CUDA backend, `-std=c++17` | Device Views + host mirror + hoisted `deep_copy` |
| `managed` | CUDA backend **with SharedSpace/UVM** | `Kokkos::SharedSpace` Views — **no** `deep_copy` |
| `unified` | CUDA backend, `-std=c++17` | Unmanaged Views over host pointers — **no** `deep_copy` |

Kokkos selects the GPU arch via `Kokkos_ARCH_*` at configure time —
detect the actual compute capability rather than hardcoding it. See
`flavors/cpp/kokkos/build-integration.md`. If `managed` is requested
but the Kokkos install lacks `SharedSpace`, record the gap and fall
back to `separate` (do not switch silently).

**C/C++ + CUDA (nvcc, C++17):**

| `memory_mode` | Allocation in GPU code | Host↔device copies |
|---------------|------------------------|--------------------|
| `separate` | `cudaMalloc` device buffers | `cudaMemcpy`, hoisted only |
| `managed` | `cudaMallocManaged` | **None** — `cudaDeviceSynchronize` only |
| `unified` | none (use host pointers) | **None** (coherent HW required) |

`nvcc` selects the SM arch via `-gencode arch=compute_NN,code=sm_NN`
— detect the actual compute capability (`nvidia-smi
--query-gpu=compute_cap`) rather than hardcoding `sm_70`/`sm_80`. See
`flavors/cpp/cuda/build-integration.md`. `unified` mode needs a
coherent platform (HMM / Grace Hopper); if unavailable, record the gap
and fall back to `managed` (do not switch silently).

## Creating `.nsight-ai-port/`

Create `{source_dir}/.nsight-ai-port/` with these files:

### `.gitignore`

Copy from `assets/nsight-port-gitignore-template` so binary nsys
reports under `profiles/` are not committed. Text distillations
(`summary.txt`, `extraction_meta.txt`, trimmed CSVs) remain
trackable if you choose to commit them.

### `config.json`

Copy the template from `assets/config-template.json` and fill
in every field from the interview answers above:

| Field | Source |
|-------|--------|
| `schema_version` | `"MAJOR.MINOR.PATCH"` string; copy from the template, never edit it |
| `source_dir` | Q1 |
| `from_lang` | Q1b (auto-detected) |
| `to_lang`, `framework` | Q2 |
| `performance_mode` | Q3 Step 1 (`"single"` or `"sequence"`) |
| `problems[]` | Q3/Q4 — one entry per workload (`id`, `description`, scripts, `notes`, `covers`) |
| `engagement` | Q5 |
| `include_guard` | Q7 (default `USE_{FRAMEWORK}`) |
| `memory_mode` | Q8 (if flavor supports it) |
| `translation_mode` | Q9 (if flavor supports it; else `"sibling"`) |
| `build_correctness` | Q6 — `.nsight-ai-port/build_correctness.sh` (copied or authored) |
| `run_correctness` | Q6 — `.nsight-ai-port/run_correctness.sh` (copied or authored) |
| `verify_timeout` | Not asked — seconds to bound a verification run (default 600). Raise it if a legitimate correctness workload runs longer |
| `gpu_environment` | From gpu-check output |

Fields not from the interview keep template defaults where unset.

Run gpu-check **and** profile-check **after** writing config with
Q2 fields:

```bash
scripts/gpu-check.sh ${to_lang}-${framework}
scripts/profile-check.sh ${to_lang}-${framework}
```

If `profile-check.sh` warns that CPU sampling prerequisites are
not met, report to the user before Round 0 (see nsys profiling
guide). Do not start Round 0 until the user fixes the
environment or confirms they will address sampling first.

### `progress.json`

Use the template from `assets/progress-template.json`. Set:
- `status`: `"in_progress"`
- `last_working_commit`: empty (filled after the Phase 0 commit)
- `problems`: one `"pending"` entry per `config.json:problems[]` item
- `functions`: empty array until Phase 1 plan approval (see
  below)

#### Top-level schema

| Field | Type | Purpose |
|-------|------|---------|
| `status` | string | `"in_progress"` or `"completed"` |
| `last_working_commit` | string | Git SHA after last verified commit; rewrite after every one |
| `current_problem` | string | Sole active id; resolves through `config.json:problems[]` |
| `current_round` | number | Active round (0 before the first round) |
| `rounds` | array | Cumulative per-round history (see below) |
| `profile_rounds` | array | Cumulative nsys profile history |
| `setup_complete` | bool | Phase 0 done |
| `plan_complete` | bool | Phase 1 done (for the active problem) |
| `refactor_complete` | bool | Phase 2 done (or skipped) for the active problem |
| `device_utils_complete` | bool | Phase 3 done (or skipped) for the active problem |
| `functions` | array | Per-function translation progress (seeded in Phase 1) |
| `translated_functions` | array | `"file:function"` specs completed — **cumulative across problems and runs** |
| `problems` | array | Per-problem progress — **cumulative across runs** (see below) |
| `notes` | string | Free-form porting notes |

The per-problem `plan`/`refactor`/`device_utils` flags, `functions[]`,
and `current_round` describe the **active** problem. When a problem
completes and the next one begins (sequence mode) or a new problem is
appended (a later run), reset those active fields and set
`current_problem`. The `problems[]`, `rounds[]`, `profile_rounds[]`,
and `translated_functions` lists remain cumulative.

To activate a problem, set `current_problem` to its id and resolve its
configuration from the matching `config.json:problems[]` entry. Never
infer the active problem from list order or profile paths. Build and run
its performance pair once and record the CPU baseline before starting
its first round. Setup performs this activation for the first problem;
later problems are activated one at a time by the outer sequence loop.

#### `problems[]` entry schema (progress)

One entry per configured problem, appended when the problem is added.
Entries are cumulative — never overwrite or remove prior entries.

```json
{
  "id": "case_a",
  "description": "First representative workload",
  "status": "completed",
  "cpu_baseline_seconds": 9.0,
  "rounds_completed": 3,
  "functions_ported": ["src/module.f90:kernel_a"],
  "last_working_commit": "a1b2c3d"
}
```

| Field | When set |
|-------|----------|
| `id`, `description` | When added — from `config.json:problems[]` |
| `status` | `"pending"` → `"in_progress"` on activation → `"completed"` when the user stops iterating on it |
| `cpu_baseline_seconds` | On activation, after the CPU performance run |
| `rounds_completed` | Incremented at each continue gate for this problem |
| `functions_ported` | `"file:function"` specs completed while this problem was active |
| `last_working_commit` | Git SHA at each continue gate and when the problem stops |

#### `functions[]` entry schema

Seed one object per translation target when the user approves
`porting_plan.json` (Phase 1). Do **not** pre-populate in the
template — add entries from the plan only.

```json
{
  "file": "src/physics/solver.f90",
  "function": "compute_kernel",
  "status": "pending",
  "phases": {
    "language_translate": false,
    "build_explore": false,
    "scaffold": false,
    "accelerate": false,
    "verify_reference": false,
    "integrate": false
  },
  "test_dir": "src/physics/openacc/test/test_compute_kernel",
  "test_executable": "",
  "commit_hash": ""
}
```

| Field | When set |
|-------|----------|
| `file`, `function` | Phase 1 — copied from `porting_plan.json` |
| `status` | `"pending"` at seed; `"completed"` after Phase 4e |
| `phases.language_translate` | Stage A (cross-language only; the transition file completed). Always `false`/ignored when `from_lang == to_lang` |
| `phases.build_explore` | Phase 4a (reuse `.nsight-ai-port/build_notes.txt` counts) |
| `phases.scaffold` | Phase 4b — also set `test_dir`, `test_executable` |
| `phases.accelerate` | Phase 4c (all OpenACC chunks done) |
| `phases.verify_reference` | Phase 4d (numerical test passes) |
| `phases.integrate` | Phase 4e — also update `commit_hash`, append to `translated_functions` |

If the user edits the plan after seeding, **or a later round
re-plans**, sync `functions[]` by `file`+`function`: add new
targets, drop non-completed entries no longer in the plan,
preserve phase flags and `commit_hash` only for `completed`
identities, and reset other retained entries to pending phase
state. Do not append a second object for a function that is
already listed.

#### `rounds[]`

Append one object at the continue gate at the end of every round:

```json
{
  "round": 1,
  "problem": "main",
  "functions_ported": ["src/foo.f90:bar"],
  "assess": {"gpu_active_pct": 61.4, "whitespace_pct": 38.6},
  "user_continued": null
}
```

| Field | Type | Purpose |
|-------|------|---------|
| `round` | number | Round number (matches `current_round`) |
| `problem` | string | Active problem id (required in sequence mode; optional in single) |
| `functions_ported` | array | `"file:function"` specs completed this round |
| `assess` | object | Metrics: gpu_active_pct / whitespace_pct |
| `user_continued` | bool \| null | Gate outcome: `true`=continued, `false`=stopped; `null` until answered |

#### `profile_rounds[]`

Append one object after each iteration report. Sequence-mode example:

```json
{
  "round": 1,
  "problem": "case_a",
  "profile_dir": ".nsight-ai-port/profiles/case_a/round_1",
  "report_path": ".nsight-ai-port/profiles/case_a/round_1/gpu.nsys-rep",
  "functions_ported": ["src/foo.f90:bar"],
  "user_continued": true
}
```

In `sequence` mode, tag each entry with the active `problem` id and
namespace `profile_dir` under it
(`.nsight-ai-port/profiles/{problem_id}/round_1`). Set `current_round`
to the active round number in `progress.json`.

See
[planning-from-trace.md](../orchestrators/nsys/references/planning-from-trace.md)
for the seed step.

### `skill_feedback.json`

Copy the template from
`assets/skill_feedback-template.json`. Leave `entries` and
`usage_log` empty and `session_summary` null until they are needed.
Append a first `usage_log` snapshot before the Phase 0 commit, then
before every subsequent commit. See
[skill-feedback-log.md](skill-feedback-log.md) and
[session-retrospective-template.md](session-retrospective-template.md).

## Proposing Verification Scripts

If the user does not have existing scripts, examine the build
system and propose them.

Write all scripts **into `.nsight-ai-port/`** (never at the source-tree
root or outside it), record their paths in `config.json`, and
`chmod +x` them. You author:

- `.nsight-ai-port/build_correctness.sh` / `.nsight-ai-port/run_correctness.sh`
  — the numerical correctness gate (Q6).
- Performance scripts as specified in Q4.

The build pattern below applies to all build scripts.

### Build script pattern

The build script **must read `USE_OPENACC` from its own
environment** and translate it into whatever the build system
needs. `USE_OPENACC` is passed directly to the script as a
prefix variable (e.g. `USE_OPENACC=1 bash .nsight-ai-port/build_correctness.sh`),
NOT set via `env.sh`.

**Make-based example:**

```bash
#!/usr/bin/env bash
set -e
cd "$NSIGHT_AI_PORT_SOURCE_DIR"

# Forward the one switch and let the Makefile decide everything else.
# The Makefile owns the compiler paths, the per-mode -gpu=…,mem:… flags,
# -Mpreprocess, and the per-configuration object tree — pinned with `:=`
# so a stale ambient FC cannot silently take over the build.
make -j USE_OPENACC="${USE_OPENACC:-0}"
```

Resist the temptation to assemble `FC=… FFLAGS="…"` here instead. Flags
spelled out on the command line have to be repeated identically by every
caller — the correctness build, the performance build, the test targets,
and you, by hand, during debugging — and the first one that differs
produces a build nobody can explain. Written into the Makefile, the rule
is applied the same way every time and shows up in code review.

**CMake-based example:**

```bash
#!/usr/bin/env bash
set -e

# One binary dir per configuration: sharing one means the second
# configuration silently reuses the first one's cache.
if [ "${USE_OPENACC:-0}" = "1" ]; then
  BUILD_SUBDIR="gpu"; CMAKE_DEFINES="-DUSE_OPENACC=ON"
else
  BUILD_SUBDIR="cpu"; CMAKE_DEFINES=""
fi

cmake -S "$NSIGHT_AI_PORT_SOURCE_DIR" \
      -B "$NSIGHT_AI_PORT_BUILD_DIR/$BUILD_SUBDIR" \
      $CMAKE_DEFINES
cmake --build "$NSIGHT_AI_PORT_BUILD_DIR/$BUILD_SUBDIR" -j
```

Key rules:
- The script must be **idempotent** (safe to run repeatedly).
- It must exit non-zero on failure.
- It should compile the **entire** project, not just one target.
- The script reads `USE_OPENACC` (or the project's custom include guard)
  from its environment and translates it into the build system's switch.
  That translation lives in the build script, NOT in `env.sh`.
- Compiler and flag **decisions** live in the Makefile/CMakeLists, not in
  the script and not in `env.sh` — see
  [verification-guide.md](verification-guide.md#the-build-system-owns-compiler-choice).
- CPU and GPU must write to different directories.

### Run script pattern

The run script executes the project's test suite or a
representative workload:

```bash
#!/usr/bin/env bash
set -e
cd "$NSIGHT_AI_PORT_SOURCE_DIR"
./my_program < test_input.dat > run_stdout.txt

# Extract the DATA lines, then compare those numerically — never diff raw
# stdout (see "Golden references" below). Here data rows are the numeric
# ones; status lines, including the NSIGHT_AI_PORT_* markers, are dropped.
grep -E '^[[:space:]]*-?[0-9]' run_stdout.txt > run_data.txt || true

python3 - run_data.txt .nsight-ai-port/gold/main/output.dat <<'PY'
import sys
cand, ref = ([[float(x) for x in l.split()]
              for l in open(p)] for p in sys.argv[1:3])
flat = lambda rows: [v for r in rows for v in r]
a, b = flat(cand), flat(ref)
if len(a) != len(b):
    sys.exit(f"FAIL: {len(a)} values vs {len(b)} in reference")
num = sum(abs(x - y) for x, y in zip(a, b))
den = sum(abs(y) for y in b)
wmad = num / den if den else num
print(f"wmad {wmad:.3e}  n {len(a)}")
sys.exit(0 if wmad <= 1e-12 else f"FAIL: wmad {wmad:.3e}")
PY
```

The `|| true` on `grep` is load-bearing: see
[Shell pitfalls under `set -e`](#shell-pitfalls-under-set--e).

Key rules:
- Must exit 0 on success, non-zero on failure.
- Should run quickly (under 10 minutes ideally).
- Must produce **deterministic** output for correctness checks.
- Never diff raw stdout against a golden file — see below.

### Golden references are data files, not stdout

A correctness gate that does `diff <program stdout> expected_output.txt`
breaks as soon as the skill does its job, for two independent reasons:

1. **The skill adds lines to stdout on purpose.** The execution markers
   (`NSIGHT_AI_PORT_LANG_EXEC`, `NSIGHT_AI_PORT_FRAMEWORK_EXEC`) are required
   proof that the translated and GPU paths actually ran, and they are
   printed to stdout because they are status, not errors. For a program
   whose scientific result *is* its stdout, a raw diff then reports a
   spurious failure on every accelerated run.
2. **Text-formatted floating point makes `diff` a bit-exactness gate.**
   Two numerically fine builds print `0.30000000000000004` versus
   `0.3` and `diff` calls it a regression. A GPU port will never pass
   that, and loosening it is not possible — `diff` has no tolerance.

So set the gate up this way:

- **Capture a golden data reference once, from the pristine baseline,
  before any source modification**, and store it under
  `.nsight-ai-port/gold/{problem_id}/`. Commit it. This is the only
  moment the original behavior is available untouched.
- **Prefer a data artifact over stdout.** If the program can write results
  to a file, compare that file. Reserve stdout for status.
- **If results only ever go to stdout**, capture stdout to a file and have
  the comparison **extract the data lines** — dropping agent-injected
  marker lines and anything inherently variable (timings, host names, step
  timers, thread counts). Filter by pattern, and prefer selecting the data
  you want over blacklisting noise you happened to notice.
- **Compare numerically, with tolerance.** Parse the numbers and apply the
  same `wmad`/tolerance approach used elsewhere, rather than `diff`. See
  [verification-guide.md](verification-guide.md#comparing-numerical-results).

Keep the markers on stdout. They are cheap, they are the primary defense
against silent CPU fallthrough, and the comparison — not the program —
is the right place to know which lines are data.

### Shell pitfalls under `set -e`

Correctness and performance scripts run under `set -e` (or
`set -euo pipefail`), where a command that returns non-zero aborts the
script. Several tools return non-zero on the **success** path:

| Command | Returns non-zero when | Why it bites |
|---|---|---|
| `grep` / `grep -c` / `grep -vc` | **nothing matched** | Counting mismatches: zero mismatches is the *good* outcome, and it kills the script |
| `diff` | files differ | Fine as a gate, wrong when you only wanted the text |
| `<compiler> -print-file-name=` | library off the default search path | Probing for an optional library |
| `wc -l` on a failed pipeline | upstream failed | Silent zero counts with `pipefail` |

**Rule: any command whose empty or negative result is an acceptable
outcome must be written so it cannot abort the script.** Append
`|| true`, or capture the status explicitly:

```bash
# WRONG — aborts precisely when every row is bit-exact
n_bad=$(grep -vc '^OK' compare.log)

# RIGHT — zero mismatches is success, not an error
n_bad=$(grep -vc '^OK' compare.log || true)
n_bad=${n_bad:-0}

# RIGHT — when you need to distinguish "no match" from "grep failed"
if n_bad=$(grep -vc '^OK' compare.log); then :; elif [ $? -gt 1 ]; then
  echo "grep failed" >&2; exit 1
else
  n_bad=0
fi
```

This failure mode is especially misleading because the script dies on the
**passing** path, which presents as a correctness regression in whichever
configuration was actually working. If a configuration fails only when its
numbers are perfect, suspect this before suspecting the numerics.

### Presenting to the user

Show the proposed scripts and say:
> "I've drafted build and run scripts based on your project's
> build system. Please review them and make any edits. I need
> your approval before proceeding."

Do **not** proceed until the user approves or provides their own.

## Performance Scripts

Each problem's `run_performance` script runs **one** representative
workload. Nsys profiles it. It is **separate** from
the correctness scripts. Q4 is the source of truth for script naming,
paths, and ownership.

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "${NSIGHT_AI_PORT_SOURCE_DIR:?}"
# Exactly ONE case — no ctest, no loop over cases, no make test
./my_program --case smoke --steps 10 < inputs/smoke.deck
```

Rules:

- **One program invocation** per script execution (no shell loop
  over multiple inputs). A **sequence** is expressed as multiple
  scripts, never as a loop inside one script.
- Must match that problem's `description`.
- Must exit with the problem's `expected_exit_code` (default 0).
- Must run **long enough to profile** — at least ~1 s of wall time. Time it
  before accepting it (Q4 Step 3); a millisecond-scale smoke case yields
  an empty trace.
- Should finish in under ~10 minutes if possible.
- Record the paths in each `problems[].build_performance` /
  `problems[].run_performance`.
- Present proposed scripts to the user for approval.

Resolve the active entry using `progress.json:current_problem`. For
example, after assigning its paths to `P_BUILD_PERFORMANCE` and
`P_RUN_PERFORMANCE`, nsys wraps the run as:

```bash
source .nsight-ai-port/env.sh && bash "$P_RUN_PERFORMANCE"
```

Honor the active entry's `working_dir` when non-empty.

## Environment Script

After verification scripts are established, write
`.nsight-ai-port/env.sh`. This script sets **static** environment
only — paths, library locations, compiler discovery. It must
NOT contain any build-mode flags (`USE_OPENACC`, cmake defines,
`-acc`, etc.). Build-mode logic belongs in the build scripts.

```bash
#!/usr/bin/env bash
# .nsight-ai-port/env.sh — static environment for verification.
# Source this before running the build/run scripts.
# NEVER set USE_OPENACC or build-mode flags here.

export NSIGHT_AI_PORT_SOURCE_DIR="/absolute/path/to/project"
export NSIGHT_AI_PORT_BUILD_DIR="/absolute/path/to/project/build"
```

Fill in paths from `config.json`. Rules:

- `NSIGHT_AI_PORT_SOURCE_DIR` = `config.json:source_dir`
- `NSIGHT_AI_PORT_BUILD_DIR` = wherever the build system puts
  artifacts (often `source_dir/build`). If the project builds
  in-source, omit or set to the same as source_dir.
- Treat `NSIGHT_AI_PORT_BUILD_DIR` as a **root** with one subdirectory per
  configuration — `$NSIGHT_AI_PORT_BUILD_DIR/cpu` and `/gpu`. The CPU and
  GPU builds must never share an object tree or a CMake binary dir, or
  the linked binary can end up part CPU and part GPU. See
  [verification-guide.md](verification-guide.md#per-configuration-build-trees).
- Add any static paths the build needs (e.g. library paths,
  CUPTI path for `NV_ACC_TIME`).
- Put **static paths** here, never compiler *choices*. Which compiler
  builds CPU versus GPU belongs in the Makefile/CMakeLists, pinned with
  `:=` — see
  [verification-guide.md](verification-guide.md#the-build-system-owns-compiler-choice).

Mark the file executable: `chmod +x .nsight-ai-port/env.sh`

### Usage pattern

Build-mode flags (`USE_OPENACC`) are passed **directly** to the
build/run command as a prefix variable. This is critical — do
NOT pass them to `source env.sh` because bash unwinds prefix
variables after the `source` builtin returns.

```bash
# CPU correctness:
source .nsight-ai-port/env.sh && bash .nsight-ai-port/build_correctness.sh
source .nsight-ai-port/env.sh && bash .nsight-ai-port/run_correctness.sh

# GPU correctness:
source .nsight-ai-port/env.sh && \
  USE_OPENACC=1 bash .nsight-ai-port/build_correctness.sh
source .nsight-ai-port/env.sh && \
  USE_OPENACC=1 bash .nsight-ai-port/run_correctness.sh
```

The performance pair (`build_performance.sh` / `run_performance.sh`)
follows the same prefix convention.

**Why this order?** `USE_OPENACC=1 bash .nsight-ai-port/build_correctness.sh`
passes the variable as a prefix to an external command (`bash`), so it
is exported into that process normally and is visible to the build
script and everything it invokes. Placing it before `source` does NOT
work — bash unwinds it when `source` returns.

If the project uses a custom include guard (e.g. a project-prefixed
`MYPROJ_OPENACC`), replace `USE_OPENACC` with that guard in
the usage pattern and in the build scripts.

## Baseline Test

Once scripts and env.sh are ready:

1. Run the correctness build:
   ```bash
   source .nsight-ai-port/env.sh && bash .nsight-ai-port/build_correctness.sh
   ```
   Must exit 0.
2. Run the correctness suite (`run_correctness`):
   ```bash
   source .nsight-ai-port/env.sh && bash .nsight-ai-port/run_correctness.sh
   ```
   Must exit 0.
3. Activate only the first problem: set
   `progress.json:current_problem` to its id, build its performance
   script, set its progress status to `"in_progress"`, and run it once
   on CPU:
   ```bash
   source .nsight-ai-port/env.sh && bash "$P_BUILD_PERFORMANCE"
   source .nsight-ai-port/env.sh && bash "$P_RUN_PERFORMANCE"
   ```
   It must exit with the first problem's `expected_exit_code`. Record
   its wall time in `progress.json:problems[].cpu_baseline_seconds`.
   Do not baseline other problems yet; the outer sequence loop
   activates and baselines exactly one problem at a time.
4. Confirm the scripts do not modify tracked source files
   (run `git status` — should be clean or only show the new
   `.nsight-ai-port/` directory).

If baseline fails, do **not** proceed. Ask the user to fix the
build/run environment first.

## Git Commit

Stage and commit `.nsight-ai-port/`:
```bash
git add .nsight-ai-port/
git commit -m "nsight-port: initialize porting configuration"
```

Mark `setup_complete: true` and record `git rev-parse HEAD` (this
commit) as `last_working_commit` in `progress.json`. An earlier SHA
would make a rollback delete `.nsight-ai-port/`.

## Archiving Runs

The skill keeps a project **composable across invocations**: finishing
a port and later running it again (to add a problem, whether the first
run was `single` or `sequence`) must not clobber or confuse the earlier
run's state. Archiving is what makes that safe.

**Archive when:**

- The port wraps up (`status: "completed"` at Phase 6), **and**
- A completed project is reconciled to **add problem(s)** or **start
  over** (see "Start-of-Run Reconciliation").

**How:** snapshot the finished run into a timestamped directory and retain
every prior-run snapshot.

```bash
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p .nsight-ai-port/history/$TS
cp .nsight-ai-port/progress.json .nsight-ai-port/history/$TS/progress.json
# Copy profile text distillations if present (binary .nsys-rep are gitignored):
[ -d .nsight-ai-port/profiles ] && cp -r .nsight-ai-port/profiles \
  .nsight-ai-port/history/$TS/profiles 2>/dev/null || true
```

**What survives an archive (the composable state):**

- **Git history** and the ported code — untouched.
- `config.json` — the living config; the reconciliation flow appends
  new problems to `problems[]` rather than replacing them.
- `progress.json:translated_functions` and `progress.json:problems[]`
  — **cumulative**; they carry forward so a new problem's plan skips
  already-ported functions and shows them as done.
- `.nsight-ai-port/history/{timestamp}/` — the snapshot of each finished
  run for audit/inspection.

**What resets for new work (add-problem / start-over):** the *active*
pointers — `current_problem`, `current_round`,
`plan_complete` / `refactor_complete` / `device_utils_complete`, and
`functions[]` (the active problem's per-function progress). Resetting
these is what prevents a stale `current_*` from confusing the agent on
the next run. `status` returns to `"in_progress"`.

Binary artifacts under `history/` are gitignored (see the gitignore
template); commit the text snapshots if you want them tracked.
