# Refactoring Guide (Phase 2) — C/C++

Language docs: `flavors/cpp/`. Subskill `refactor-task` always
reads this file (Phase 2 edits **source** C++). When
`from_lang == to_lang`, also read the framework addendum
`flavors/cpp/{framework}/refactoring.md`.

## When to Refactor

Refactor when a function that should be GPU-accelerated is tangled
with code that cannot run on a GPU or cannot be tested in isolation:

1. **Orchestration + compute mixed.** A function both manages workflow
   (file I/O, MPI, allocation) and contains the hot compute loop.
   Extract the compute into a new free function with flat arguments.

2. **CPU thread regions spanning multiple calls.** A host threading
   region (`#pragma omp parallel`, `std::thread`, a thread pool) wraps
   several function calls. Extract the parallel region into a
   standalone function so a GPU kernel can replace it.

3. **Host code between parallel regions.** MPI communication, file
   I/O, or runtime-library calls appear between two parallelizable
   loop nests. Extract each loop nest into its own function so each
   becomes a clean GPU kernel.

4. **Widely separated loop dimensions.** An outer loop in a caller
   invokes a callee whose loops cover different dimensions (e.g. caller
   loops over `k`, callee loops over `j,i`). Do NOT list both as
   translation targets. Refactor the caller into strided blocks and
   pass block bounds to the callee so one target owns all the parallel
   dimensions. See "Strided block refactor" below.

5. **Compute trapped in a class method using members.** The target is
   a private/non-exposed method that reads many `this->` members and
   cannot be called from a standalone test. Extract a free function
   (or `static` method) taking the needed data as **explicit flat
   arguments** (pointers + sizes), and have the method forward to it.
   This also keeps objects out of device kernels (see the
   framework addendum).

6. **Device-unsafe argument types on GPU targets.** A function takes
   heavyweight host types (deep STL containers of containers,
   polymorphic objects, types with virtual methods) and dereferences
   them inside hot loops. Extract a GPU-callable entry point that takes
   flat array pointers + extents. The host keeps the rich-type
   orchestrator; the kernel/wrapper receives raw `.data()` pointers.
   See "Decoupled GPU Entry Point" below.

## Refactoring Rules

**DO:**
- Make structural changes only (move code, change signatures)
- Keep the same program behavior — outputs must not change
- Use standard C++ (no GPU framework syntax)
- Keep extracted GPU-target signatures as **flat pointers +
  scalars** so a later GPU wrapper can be declared without
  framework types in the host header (CUDA addendum: plain C++
  header for the `.cu` wrapper)
- Update all callers when changing function signatures
- Update `#include` statements and declarations for moved code

**DO NOT:**
- Add any GPU or framework code
- Change algorithms or numerical behavior
- Introduce new dependencies
- Break the build at any point

## Per-Task Workflow

For each task in `porting_plan.json["refactor_tasks"]`:

1. **Read the instructions** from the plan item.
2. **Read the target function** and understand the code flow.
3. **Make the change.** Typical patterns:
   - Extract function: move the loop body into a new free function
     with explicit flat arguments, add a call at the extraction point.
   - Move/declare: add the new function to a header so the test and
     callers can reach it.
4. **Run build verification** (build script must pass).
5. **Run test verification** (run script must pass — behavior
   unchanged).
6. **Git commit:**
   ```bash
   git add -A
   git commit -m "refactor: extract {new_func} from {old_func}"
   ```
7. **Update the plan** if new functions were created:
   - **Orchestration + compute extraction** (cases 1–3): add the
     extracted compute function to `functions[]` if it is now the
     clean GPU target. The original orchestrator is NOT a target.
   - **Strided block refactor** (case 4): do NOT add the inner callee
     to `functions[]`. Add it to `device_utilities` instead. Only the
     caller remains in `functions[]`.
   - **Callability/class extraction** (case 5): add extracted free
     functions to `functions[]` if they are GPU targets.
   - **Decoupling extraction** (case 6): add the flat-argument GPU
     entry point to `functions[]`. The rich-type wrapper is NOT a
     translation target — it dispatches at the call site during
     integration.

## Decoupled GPU Entry Point (Device-Unsafe Types)

Use when a translation target takes heavyweight host types and the
kernel must not receive the whole object — only flat data crosses the
GPU boundary. Heavyweight host types (deep STL, virtual methods)
must not cross into the device kernel. The wrapper's public header
stays **plain C++** (flat pointers).

**Goal:** A host-callable routine with the original signature
(unchanged for CPU builds) plus a new flat-argument GPU entry point
that the host wrapper calls under `#ifdef {include_guard}`. The
suffix and guard come from the active framework pack. See that
pack's `refactoring.md` addendum for the dispatch snippet.

### Steps

1. Identify every array/scalar the hot loop reads or writes.
2. Create `{name}_{framework}(...)` taking raw pointers + integer
   extents (and lower bounds if non-zero).
3. Move the loop body into the new function (pure C++, no GPU
   framework yet — kernels come in Phase 4c).
4. The original function body becomes a thin CPU path (or the call
   site dispatches directly during integration).
5. Standalone tests call `{name}_{framework}` with the same flat
   arguments, allocating test buffers directly.

### Example

**Before:**
```cpp
void compute_interaction(const Grid& grid, const State& state,
                         std::vector<double>& du) {
  for (int k = 0; k < grid.nk; ++k)
    for (int j = grid.js; j < grid.je; ++j)
      for (int i = grid.is; i < grid.ie; ++i)
        du[idx(i,j,k)] = grid.dx[ij(i,j)] * state.h[idx(i,j,k)] /* ... */;
}
```

**After refactor (still no GPU framework — Phase 4c adds kernels):**
```cpp
void compute_interaction_gpu(const double* dx, const double* h,
                             double* du,
                             int is, int ie, int js, int je, int nk) {
  for (int k = 0; k < nk; ++k)
    for (int j = js; j < je; ++j)
      for (int i = is; i < ie; ++i)
        du[idx(i,j,k)] = dx[ij(i,j)] * h[idx(i,j,k)] /* ... */;
}
```

**Integration dispatch (Phase 4e):** use the active pack's
`include_guard` and `_{framework}` suffix. See the framework
`refactoring.md` addendum for a copy-paste snippet. Passing
`.data()` passes the underlying buffer by pointer. The `_gpu`
name above is a placeholder (`_cuda` or `_kokkos`).

## Strided Block Refactor (Separated Loop Dimensions)

Use when a translation target would iterate over fewer array
dimensions than its callers because outer loop variables live in a
caller higher up the call stack. The inner function will not expose
enough parallelism for efficient GPU execution if ported alone.

**Goal:** One translation target owns all dimensions needed for GPU
parallelism, while preserving CPU behaviour unchanged.

### Steps

1. Add a `stride` parameter to the caller (default `1`).
2. Replace the outer loop with strided blocks and pass block bounds to
   the callee.
3. The callee loops over `[start, end)` internally for each dimension
   previously owned by the caller.
4. Update the plan: caller stays in (or moves into) `functions[]`; the
   callee moves to `device_utilities[]`, not `functions[]`.
5. Document in the refactor task: CPU call sites use the default
   `stride=1`; GPU translation may use a larger stride to fuse
   dimensions across the kernel's grid.

### Example

**Before:**
```cpp
void driver(double* a3d, double* b3d, int ni, int nj, int nk,
            double dt) {
  for (int j = 0; j < nj; ++j)
    for (int i = 0; i < ni; ++i)
      column_rhs(&a3d[col(i,j)], &b3d[col(i,j)], nk, dt);
}
```

**After:**
```cpp
void driver(double* a3d, double* b3d, int ni, int nj, int nk,
            double dt, int stride = 1) {
  const int s = stride < 1 ? 1 : stride;
  for (int j0 = 0; j0 < nj; j0 += s) {
    const int j_end = std::min(j0 + s, nj);
    for (int i0 = 0; i0 < ni; i0 += s) {
      const int i_end = std::min(i0 + s, ni);
      column_rhs(a3d, b3d, ni, nj, nk, dt, i0, i_end, j0, j_end);
    }
  }
}

void column_rhs(double* a3d, double* b3d, int ni, int nj, int nk,
                double dt, int i_start, int i_end,
                int j_start, int j_end) {
  // loops i_start:i_end, j_start:j_end, then k internally
}
```

With `stride=1`, behaviour is identical to the original. With a larger
stride, the GPU translation can map all dimensions onto the kernel
grid inside `driver`.

Choose which dimensions to stride based on the specific code — not
every outer dimension needs striding. Focus on dimensions that would
provide meaningful additional GPU parallelism.

### CPU vs GPU stride

| Path | Stride | Effect |
|------|--------|--------|
| CPU (default) | 1 | Original loop structure preserved |
| GPU translation | > 1 | Fuses outer/inner dims onto the kernel grid |

Existing call sites need no changes when `stride` defaults to `1`.

## Example: Extract Compute Kernel

**Before:**
```cpp
void timestep_loop(Grid& grid, double dt, int nsteps) {
  for (int step = 0; step < nsteps; ++step) {
    exchange_halos(grid);                 // MPI — cannot run on GPU

    for (int k = 0; k < grid.nz; ++k)     // GPU candidate
      for (int j = 0; j < grid.ny; ++j)
        for (int i = 0; i < grid.nx; ++i)
          grid.rhs[idx(i,j,k)] = /* complex computation */;

    if (step % 100 == 0) write_checkpoint(grid);   // I/O
  }
}
```

**After:**
```cpp
void compute_rhs(double* rhs, const double* state,
                 int nx, int ny, int nz) {
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i)
        rhs[idx(i,j,k)] = /* complex computation */;
}

void timestep_loop(Grid& grid, double dt, int nsteps) {
  for (int step = 0; step < nsteps; ++step) {
    exchange_halos(grid);
    compute_rhs(grid.rhs.data(), grid.state.data(),
                grid.nx, grid.ny, grid.nz);
    if (step % 100 == 0) write_checkpoint(grid);
  }
}
```

Now `compute_rhs` is a clean GPU candidate with flat arguments,
no I/O, no MPI, and no host objects that would have to cross into a
kernel.

## Example: Extract from a Class Method

**Before** (compute trapped in a private method using members):
```cpp
class Solver {
  std::vector<double> state_;
  int nx_, nz_;
 public:
  void step(double dt) {
    for (int k = 0; k < nz_; ++k)
      for (int i = 0; i < nx_; ++i)
        state_[idx(i,k)] += dt * /* ... uses members ... */;
  }
};
```

**After** (free function with explicit arguments; method forwards):
```cpp
// solver_kernels.hpp  (plain C++ — future GPU wrapper header)
void advance_state(double* state, int nx, int nz, double dt);

// solver_kernels.cpp
void advance_state(double* state, int nx, int nz, double dt) {
  for (int k = 0; k < nz; ++k)
    for (int i = 0; i < nx; ++i)
      state[idx(i,k)] += dt * /* ... */;
}

class Solver {
  std::vector<double> state_;
  int nx_, nz_;
 public:
  void step(double dt) {
    advance_state(state_.data(), nx_, nz_, dt);
  }
};
```

Now `advance_state` can be called from a standalone test, takes flat
arguments, and its signature is expressible in a plain C++ header — so
the Phase 4c GPU wrapper can be declared without framework types
leaking into `Solver`'s translation unit.

## Subagent Use

Each refactor task is independent — it has:
- **Input:** the refactoring instructions + current source
- **Success criterion:** build + run scripts pass
- **Output:** git commit with the structural change

This makes each task a natural subagent boundary.
