# CUDA-Q Authoring Reference

Use this reference for `/cudaq-guide author` and for debugging Python
`@cudaq.kernel` code. It targets the CUDA-Q 0.14 and 0.15 Python APIs;
re-check version-sensitive behavior against the installed `cudaq.__version__`.
If the installed version differs from the latest documentation, review the
relevant documentation or source changes while debugging.

For porting Qiskit code to CUDA-Q, use the `cudaq-importing` skill. That skill
contains Qiskit gate mappings, Qiskit-vs-CUDA-Q bit ordering, precision
comparison guidance, framework-decoupling patterns, and port validation.

## Defaults and Disciplines

### Prefer decorator mode

Decorator mode (`@cudaq.kernel`) is the default. Do not use
`cudaq.make_kernel()` builder mode without explicit user permission.

Builder mode has uneven feature support across CUDA-Q 0.14/0.15-era Python
APIs: no `mz(list)`, no qview slicing, no `x.ctrl`, no `cudaq.adjoint` on
builder kernels, `apply_call` on subkernels can be fragile, and
`QuakeValue.__getitem__` rejects NumPy ints. Count-key ordering also differs
between modes. If a case appears to need builder mode, describe the advantage
and tradeoff and wait for confirmation.

### Avoid unnecessary restrictions

Do not introduce restrictions unless the user directs. Common mistakes:

- Hardcoded register sizes such as `cudaq.qvector(10)` when
  `cudaq.qvector(num_qubits)` with an `int` parameter works.
- Fixed-arity gate dispatchers (`if nc == 0 / 1 / 2`) that silently drop
  higher-arity multi-controlled gates. Use runtime-arity list comprehensions.
- `NotImplementedError` for flags that are no-ops in CUDA-Q, such as a
  `parameterized=True` flag. A CUDA-Q kernel is already parameterized over
  runtime float arguments.
- Per-shape kernel factories when a simpler int-parameter kernel works. Use a
  factory only when a fixed-length return-list literal is genuinely tied to a
  parameter.

If a cap is genuinely required by a CUDA-Q language constraint, raise
`NotImplementedError` with a clear message describing what is needed to lift it.
Never silently drop a gate or mis-compute.

### Validate small first

Run at the smallest valid configuration before scaling. Compare raw count keys,
not just aggregate fidelity, because bit-ordering and IQFT-direction bugs can
produce plausible-looking partial fidelities.

## Execution API Selection

Pick the API before writing the kernel. The API determines the kernel return
signature and output conventions.

| Need | API | Kernel signature | Notes |
|---|---|---|---|
| Aggregate counts only | `cudaq.sample(K, *args, shots_count=N)` | ends with `mz(qubits)`, no return type | Default and fast |
| Per-shot bits captured in Python | `cudaq.run(K, *args, shots_count=N)` | `-> List[bool]` or `-> int` | Needed for mid-circuit measurement, post-selection, or feed-forward; can be orders of magnitude slower |
| Repeated sequential measurements | `cudaq.sample(..., explicit_measurements=True)` | measurement-order bitstring | Bit order is measurement order, not allocation order |
| Statevector amplitudes | `cudaq.get_state(K, *args)` | no `mz` | Simulator-only exact distributions and fidelity baselines |
| Pauli expectation value | `cudaq.observe(K, hamiltonian, *args)` | no `mz` | Use for spin-operator expectation values |

Rule of thumb: when the kernel does not need per-shot data in Python, end with
`mz(qubits)` and omit the return type so execution uses `cudaq.sample`.

`cudaq.SampleResult` is not a full dict. It supports `.items()` and indexed key
access, but not every dict method reliably. Convert when downstream code expects
a dict:

```python
counts = {k: v for k, v in result.items()}
```

## CUDA-Q Kernel Language

### Supported inside `@cudaq.kernel`

- Qubit allocation: `cudaq.qubit()`, `cudaq.qvector(n)`, and parameter type
  `cudaq.qview`.
- Qview slicing: `qubits[a:b]` with runtime `a` and `b`.
- `for` loops with `range(start, stop, step)` for positive or negative steps.
- `if` / `elif` / `else` with runtime conditions.
- Direct sub-kernel calls: `my_subkernel(args)`.
- `cudaq.control(K, ctrl_or_list, *args)` to propagate control through every
  gate in `K`.
- `cudaq.adjoint(K, *args)` at top level, with caveats below.
- Multi-control gate forms: `x.ctrl(c, t)`, `x.ctrl([c1, c2], t)`,
  `x.ctrl(c1, c2, t)`, and analogous forms for `y`, `z`, `h`, `rx`, `ry`,
  `rz`, `r1`, and `swap`.
- `mz(qubit)`, `mx(qubit)`, and `my(qubit)` captured into variables for
  mid-circuit conditions and typed returns. CUDA-Q 0.14 treats captures as
  bool-like in common Python workflows; CUDA-Q 0.15 documents them as
  measurement handles that are discriminated when used in boolean conditions or
  returned through a typed `list[bool]` / `int` kernel result.
- `mz(qview)` without return capture for sampling through `cudaq.sample`.
- `reset(qubit)`.
- Float arithmetic with mixed int/float promotion.
- Typed parameters: `int`, `float`, `bool`, `cudaq.qubit`, `cudaq.qview`,
  `List[int]`, `List[float]`, and `List[bool]`.
- List comprehensions over runtime-bound lengths, such as
  `[qubits[idx[k]] for k in range(nc)]`. This enables runtime-arity
  multi-control dispatch.

### Unsupported inside `@cudaq.kernel`

These commonly produce compiler errors:

- Explicit casts `int(x)`, `float(x)`, `bool(x)`; prefer implicit promotion.
- `reversed(range(n))`; use `range(n - 1, -1, -1)`.
- Empty list literal `[]`.
- List multiplication such as `[False] * n`.
- List `.append()` mutation.
- Annotated assignment without initializer.
- `List[List[...]]` parameter or return type; flatten to parallel arrays.
- f-strings.
- Indexing with `np.int64`; cast to Python `int` before packing parameters or
  indexing.
- `np.random.*` and other runtime-dynamic library calls. Generate randomness in
  Python outside the kernel and pass values as arguments.

## Silent-Failure and Confusing-Error Pitfalls

### `cudaq.control` does not compose with inner `cudaq.adjoint`

If kernel `K` contains `cudaq.adjoint(M, ...)`, wrapping `K` in
`cudaq.control(K, ...)` can fail with:

```text
RuntimeError: Could not successfully apply argument synth.
```

Fix: hand-roll `M`'s inverse in a separate kernel by walking gates in reverse
and negating rotation angles. Self-inverse gates such as X, Y, Z, H, CX, and
CSWAP keep the same angle and operands. RY, RX, RZ, R1, RZZ, and similar
rotations negate the angle.

### `cudaq.adjoint` can fail on qview slices or nested loops

Errors include:

```text
operand #1 does not dominate this use
'quake.extract_ref' op invalid constant index value
```

Workaround: manually unroll the inverse or write a dedicated inverse kernel.

### `cudaq.run` is much slower than `cudaq.sample`

At high shot counts, `cudaq.run` returning `List[bool]` can be roughly 1000x
slower than `cudaq.sample`. Only declare a return type when per-shot data is
genuinely needed.

### Parameter-bounded loops can trigger JIT lowering errors

If `for i in range(N)` uses a runtime `N` to index a fixed allocation, the JIT
may fail even when classical logic guarantees `N` is in range. Clamp explicitly:

```python
safe_n = bound
if safe_n > num_qubits:
    safe_n = num_qubits
for i in range(safe_n):
    h(qubits[i])
```

### NumPy ints are invalid kernel-side indices

`qubits[np.int64(0)]` raises an invalid-index error. Cast values to Python
`int` before packing `List[int]` kernel parameters.

### Count-key conventions differ by mode

In decorator mode, `mz(qview)` puts `qview[0]` leftmost in the count key. In
builder mode, count keys are ordered by qubit allocation index. Do not mix modes
within a project without carefully remapping output keys.

### `reset(q)` on noiseless statevector simulation

`reset(q)` is deterministic projection to `|0>`, not stochastic collapse.
This is usually fine, but matters for algorithms depending on stochastic
collapse plus post-selection semantics.

## Recurring Patterns

### Parallel-array gate encoding

When a kernel needs a sequence of multi-controlled gates whose control count,
control indices, control states, target, and angle all depend on runtime data,
encode the gate sequence as parallel arrays.

Use flat arrays with offsets. For gate `i`, controls live in
`ctrl_indices[ctrl_offsets[i]:ctrl_offsets[i + 1]]`, with matching
`ctrl_states`.

```python
@cudaq.kernel
def apply_A(state: cudaq.qview,
            kinds: List[int], targets: List[int], thetas: List[float],
            ctrl_indices: List[int], ctrl_states: List[int],
            ctrl_offsets: List[int]):
    L = len(kinds)
    for i in range(L):
        kind = kinds[i]
        target = state[targets[i]]
        theta = thetas[i]
        c_start = ctrl_offsets[i]
        c_end = ctrl_offsets[i + 1]
        nc = c_end - c_start

        for k in range(c_start, c_end):
            if ctrl_states[k] == 0:
                x(state[ctrl_indices[k]])

        if nc == 0:
            if kind == 0:
                h(target)
            elif kind == 1:
                x(target)
            elif kind == 2:
                ry(theta, target)
        else:
            ctrls = [state[ctrl_indices[c_start + j]] for j in range(nc)]
            if kind == 1:
                x.ctrl(ctrls, target)
            elif kind == 2:
                ry.ctrl(theta, ctrls, target)

        for k in range(c_start, c_end):
            if ctrl_states[k] == 0:
                x(state[ctrl_indices[k]])
```

The key construct is the runtime-bound list comprehension:

```python
ctrls = [state[ctrl_indices[c_start + j]] for j in range(nc)]
```

This avoids fixed-arity dispatchers and hardcoded maximum control counts. X-wrap
plus gate plus X-unwrap is safe under outer `cudaq.control`.

For an outer loop applying different subcircuits per iteration, concatenate all
per-iteration gate records in Python and keep offsets so the kernel can walk
the correct slice for each iteration.

### Contiguous-swap trick for sparse controls

When a specific intrinsic requires controls as a contiguous `qview` slice,
precompute swaps in Python and gather sparse controls into a prefix:

```python
def _gather_swaps(op_controls, num_state_qubits):
    logical_at_pos = list(range(num_state_qubits))
    pos_of_logical = list(range(num_state_qubits))
    swaps = []
    for target_pos, control in enumerate(op_controls):
        source_pos = pos_of_logical[control]
        if source_pos == target_pos:
            continue
        swaps.append((target_pos, source_pos))
        a, b = logical_at_pos[target_pos], logical_at_pos[source_pos]
        logical_at_pos[target_pos], logical_at_pos[source_pos] = b, a
        pos_of_logical[a] = source_pos
        pos_of_logical[b] = target_pos
    return swaps

@cudaq.kernel
def apply_op(state: cudaq.qview, swap_a: List[int], swap_b: List[int],
             num_swaps: int, num_ctrls: int, theta: float):
    for i in range(num_swaps):
        swap(state[swap_a[i]], state[swap_b[i]])
    ry.ctrl(theta, state[0:num_ctrls], state[num_ctrls])
    for i_rev in range(num_swaps):
        i = num_swaps - 1 - i_rev
        swap(state[swap_a[i]], state[swap_b[i]])
```

Prefer the flat-list control form unless a specific intrinsic forces contiguous
controls.

### Statevector-based expected distribution

Use `cudaq.get_state` on an unmeasured kernel when an exact distribution is
needed:

```python
@cudaq.kernel
def my_circuit_unmeasured(args...):
    # Gates only, no mz.
    pass

state = cudaq.get_state(my_circuit_unmeasured, *args)
counts = {}
for i in range(2 ** N):
    amp = complex(state[i])
    p = amp.real**2 + amp.imag**2
    if p > 1e-12:
        counts[format(i, f"0{N}b")] = p
counts = {k: round(v * num_shots) for k, v in counts.items()}
```

Pair with a measured kernel containing the same gate body plus trailing
`mz(qubits)`. Since both outputs are CUDA-Q conventions, keys line up.

### Variational loop

```python
def expectation(thetas):
    counts = cudaq.sample(my_kernel, *args, thetas, shots_count=num_shots)
    return objective_from_counts(counts)

res = scipy.optimize.minimize(expectation, init_thetas, method="COBYLA",
                              options={"maxiter": max_iter})
```

`cudaq.sample` is synchronous. If a closure captures a loop variable, bind it
with a default argument (`def expectation(thetas, _idx=i)`).

CUDA-Q has no symbolic-vs-bound distinction for kernel float arguments, so
source-framework `parameterized=True` flags are often no-ops.

### Mid-circuit measurement and post-selection

Use `cudaq.run` when per-shot measurement values must be captured:

```python
@cudaq.kernel
def my_kernel(args...) -> List[bool]:
    a = mz(ancilla)
    reset(ancilla)
    return [a, mz(qa[N - 1]), mz(qa[N - 2]), mz(qa[0])]

shots = cudaq.run(my_kernel, *args, shots_count=N)
post_selected_counts = {}
for shot in shots:
    if not shot[0]:
        continue
    key = "".join("1" if b else "0" for b in shot[1:])
    post_selected_counts[key] = post_selected_counts.get(key, 0) + 1
```

An encoded integer return can be cleaner for fixed-width feed-forward circuits:

```python
@cudaq.kernel
def my_kernel(args...) -> int:
    res = 0
    if mz(ancilla):
        res = res + (1 << N)
    for q in range(N):
        if mz(qa[q]):
            res = res + (1 << q)
    return res
```

Reference examples in the CUDA-Q source:
`docs/sphinx/examples/python/measuring_kernels.py` and
`sample_to_run_migration.py`.

### Variable-shape return lists

`@cudaq.kernel` requires return-list literals to have fixed length at
decoration time.

Preferred option: use `mz(qview)` plus `cudaq.sample` when you do not need
per-shot data in Python. This avoids fixed-length return lists and is much
faster.

Factory option: generate one kernel per return shape with `exec` plus
`linecache.cache` only when `cudaq.run` and a fixed-length `List[bool]` return
are genuinely required.

```python
import linecache

_KERNEL_CACHE = {}

def _ensure_kernel(N):
    if N in _KERNEL_CACHE:
        return _KERNEL_CACHE[N]
    return_list = ", ".join([f"mz(q[{i}])" for i in range(N - 1, -1, -1)])
    src = f"""
@cudaq.kernel
def my_kernel_N{N}(...) -> List[bool]:
    q = cudaq.qvector({N})
    return [{return_list}]
"""
    fname = f"<dynamic:my_kernel_N{N}>"
    linecache.cache[fname] = (
        len(src), None, src.splitlines(keepends=True), fname)
    code = compile(src, fname, "exec")
    namespace = {"cudaq": cudaq, "List": List}
    exec(code, namespace)
    K = namespace[f"my_kernel_N{N}"]
    _KERNEL_CACHE[N] = K
    return K
```

Registering the source in `linecache.cache` lets `inspect.getsource` find the
dynamic kernel source during CUDA-Q decoration.

### Hand-rolled inverse

Use hand-rolled inverses when a kernel may be controlled:

```python
@cudaq.kernel
def A_op(state: cudaq.qview, S: int, theta: float):
    ry(theta, state[0])
    for i in range(S):
        x.ctrl(state[0], state[i + 1])

@cudaq.kernel
def A_op_inv(state: cudaq.qview, S: int, theta: float):
    for i in range(S):
        x.ctrl(state[0], state[i + 1])
    ry(-theta, state[0])

@cudaq.kernel
def Q_op(state: cudaq.qview, S: int, theta: float):
    A_op_inv(state, S, theta)

@cudaq.kernel
def main(...):
    cudaq.control(Q_op, count_ctrl, state, S, theta)
```

Inverse-emission rules:

- Walk gates in reverse order.
- X, Y, Z, H, CX, CY, CZ, SWAP, and CSWAP are self-inverse.
- RX, RY, RZ, R1, RZZ, RXX, RYY, and controlled variants negate the angle.
- S becomes `S.adj`; T becomes `T.adj`.

## Resource Metrics

`cudaq.estimate_resources(K, *args)` returns a resources object. Across
CUDA-Q 0.14/0.15, `count()` reports total gate count and
`count_controls(gate_name, num_controls)` reports gate invocations with exactly
that many controls. Newer builds may also expose arity-indexed fields such as
`gate_count_by_arity`, `gate_count_for_arity`, `depth_for_arity`,
`multi_qubit_gate_count`, and `multi_qubit_depth`; prefer those when present
and fall back to `count_controls` for older installations.

```python
resources = cudaq.estimate_resources(K, *args)
total_gates = resources.count()
controlled_gates = ["x", "y", "z", "r1", "rx", "ry", "rz"]
two_q_weighted = 0
for gate in controlled_gates:
    for arity in range(1, num_qubits):
        two_q_weighted += arity * resources.count_controls(gate, arity)
```

Interpretation caveats:

- A multi-controlled gate may count as one high-level operation or as many
  effective two-qubit controls after decomposition. Define the metric before
  comparing tools.
- Resource counters can have IR-shape blind spots. Verify on small known
  examples before treating a count of zero as real.
- `CUDAQ_TIMING_TAGS=5` enables backend-level instrumentation such as
  `Gate Count` and `Control Count`. These totals are aggregate across all
  kernel executions in a process.
- Native timing/control totals may be absent for some kernel shapes. Treat
  absence as an instrumentation quirk, not zero work.

Correctness and output fidelity are the primary validation targets; resource
counts are secondary diagnostics.

## Workflow and Debugging

General workflow:

1. Read the algorithm and identify register layout, qubit order, gate sequence,
   control/inversion structure, and expected count-key format.
2. Pick the execution API before writing the kernel.
3. Move complex Python out of kernels. Precompute angles, operation records,
   control lists, swap schedules, and parameter bindings in normal Python.
4. Probe kernel-language features with a small file in `/tmp/probe.py`; CUDA-Q
   source introspection means `python3 -c` and stdin scripts can fail at
   decoration time.
5. Run the smallest valid case first and compare raw keys or analytic outputs.
6. If a lowering error depends on input size or value, split the path into a
   dedicated kernel.

Debugging checklist:

- Counts correct but keys reversed: fix allocation, measurement, or formatting
  at the boundary.
- Fidelity around 0.4-0.6 with mass on shifted neighbors: suspect IQFT
  direction, QPE qubit reversal, or a wrong controlled-gate flavor such as
  `rz.ctrl` vs `r1.ctrl`.
- Fidelity exactly 0.0: often endianness. Verify with a 2- or 3-qubit
  deterministic probe.
- Only some widths or values fail: inspect integer arithmetic inside the kernel
  and try precomputing the offending values in Python.
- `RuntimeError: Could not successfully apply argument synth`: a controlled
  kernel probably contains `cudaq.adjoint`; hand-roll the inverse.
- `'quake.extract_ref' op invalid constant index value`: a loop bound could
  exceed allocated qvector size; clamp explicitly.
- `AttributeError` on `result.keys()`: convert `SampleResult` to a dict.
- Mid-circuit output looks wrong: compare
  `sample(..., explicit_measurements=True)` with a `cudaq.run` kernel returning
  encoded per-shot values.
- Sparse-control dispatch needs dynamic lists: use the flat-list pattern or the
  contiguous-swap schedule instead of builder mode.

## Validating a New Algorithm

For new algorithms with no source-framework reference:

1. Pick a small case with a known analytic answer.
2. Verify the output bitstring matches the analytic expectation at 100%
   probability on a noiseless simulator.
3. Scale up only after the small case is correct.
4. Use `cudaq.get_state` to inspect intermediate distributions when feasible.

When task accuracy depends on a specific CUDA-Q version, check
`cudaq.__version__`, the latest official CUDA-Q documentation, and any relevant
documentation or source changes between the installed version and latest.

## External References

1. CUDA-Q latest documentation: <https://nvidia.github.io/cuda-quantum/latest/>.
2. CUDA-Q examples for `cudaq.run`, `List[bool]`, and mid-circuit measurement:
   `docs/sphinx/examples/python/measuring_kernels.py` and
   `sample_to_run_migration.py` in the CUDA-Q source tree.
3. CUDA-Q Academic examples: <https://github.com/NVIDIA/cuda-q-academic>.
4. CUDA-Q source-tree tests: `python/tests/kernel/test_kernel_features.py` and
   `python/tests/kernel/test_measure_handle.py` when present.
