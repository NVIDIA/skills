# Qiskit to CUDA-Q Porting Reference

Use this reference when the top-level `SKILL.md` says detailed gate, ordering,
precision, or framework-decoupling guidance is needed.

## Porting Disciplines

Preserve the source algorithm when porting.

- Do not change the high-level quantum algorithm without explicit permission.
  Examples: removing mid-circuit measurements, replacing iterative QPE with a
  coherent unroll, or changing oracle decompositions.
- Do not introduce restrictions that the source implementation did not have.
- If a CUDA-Q language constraint genuinely requires a cap or unsupported path,
  raise `NotImplementedError` with a clear reason. Never silently drop gates.
- A CUDA-Q port must not depend on Qiskit, Qiskit Aer, Qiskit IBM Runtime, or
  another source framework at runtime.

Avoid these failure modes:

- Hardcoded register sizes such as `cudaq.qvector(10)` when
  `cudaq.qvector(num_qubits)` works.
- Fixed-arity dispatchers that silently ignore controls beyond a hand-coded
  limit. Use runtime-arity control lists.
- Rejecting source flags that are no-ops in CUDA-Q, such as a
  `parameterized=True` flag when the CUDA-Q kernel already accepts runtime
  parameters.
- Per-shape kernel factories when a runtime integer parameter is enough.

Framework-free ports:

- Do not `import qiskit` from CUDA-Q modules.
- Do not import a Qiskit module just to reuse pure-Python helpers.
- Do not build a `QuantumCircuit` and inspect `qc.data` at runtime.
- Move pure-Python analyzers, generators, and post-processing helpers into a
  sibling module using only stdlib and NumPy.

## Gate Translation Table

| Qiskit | CUDA-Q decorator-mode equivalent | Notes |
|---|---|---|
| `qc.h(q)`, `x`, `y`, `z`, `s`, `t`, `sdg`, `tdg` | `h(q)`, `x(q)`, `y(q)`, `z(q)`, `s(q)`, `t(q)`, `s.adj(q)`, `t.adj(q)` | Direct |
| `qc.rx/ry/rz(theta, q)` | `rx(theta, q)`, `ry(theta, q)`, `rz(theta, q)` | Direct |
| `qc.p(theta, q)` | `r1(theta, q)` | Both are `diag(1, exp(i theta))` |
| `qc.cx(c, t)` | `cx(c, t)` or `x.ctrl(c, t)` | Direct |
| `qc.cy/cz(c, t)` | `y.ctrl(c, t)`, `z.ctrl(c, t)` | Direct |
| `qc.ch(c, t)` | `h.ctrl(c, t)` | Direct |
| `qc.crx/cry/crz(theta, c, t)` | `rx.ctrl(theta, c, t)`, `ry.ctrl(...)`, `rz.ctrl(...)` | Control and target order matters |
| `qc.cp(theta, c, t)` | `r1.ctrl(theta, c, t)` | Controlled phase; symmetric mathematically |
| `qc.u/UGate/u3(theta, phi, lam, q)` | `u3(theta, phi, lam, q)` when available; otherwise decompose to rotations | Check the installed CUDA-Q version |
| `qc.cu/CUGate/cu3(theta, phi, lam, gamma, c, t)` | `u3.ctrl(theta, phi, lam, c, t)` when available, with global-phase handling if needed; otherwise decompose | Do not assume a separate `cu3` helper exists |
| `qc.rzz(theta, i, j)` | `cx(q[i], q[j]); rz(theta, q[j]); cx(q[i], q[j])` | Decomposition works across 0.14/0.15; prefer native `rzz` only if available in the installed API |
| `qc.rxx(theta, i, j)` | H on both qubits, RZZ decomposition, H on both qubits | |
| `qc.ryy(theta, i, j)` | Sdg and H basis changes around RZZ decomposition | |
| `qc.mcx([c...], t)` | `x.ctrl([c...], t)` | Arbitrary arity |
| `qc.mcry(theta, [c...], t)` | `ry.ctrl(theta, [c...], t)` | Arbitrary arity |
| `qc.mcp(theta, [c...], t)` | `r1.ctrl(theta, [c...], t)` | Preferred for many-control phase |
| `MCXGate(..., ctrl_state="010")` | X-wrap open controls before and after the controlled operation | No Qiskit-style `ctrl_state` argument; `~control` syntax may be available in the installed CUDA-Q version |
| `qc.swap(a, b)` | `swap(a, b)` | Direct |
| `qc.cswap(c, a, b)` | `swap.ctrl(c, a, b)` | Also accepts list/variadic controls |
| Final `qc.measure(q, c)` | `mz(qubits)` and `cudaq.sample` | See bit-ordering guidance |
| Mid-circuit `qc.measure(q, c)` | `b = mz(q)` plus `cudaq.run` and typed return | Use when the measurement result drives control flow or must be returned; newer CUDA-Q documents `mz` captures as measurement handles that discriminate in boolean contexts or typed returns |
| `qc.reset(q)` | `reset(q)` | Unconditional reset |
| `qc.append(U, qubits)` | Implement `U` as a CUDA-Q kernel/function and call it | |
| `qc.inverse()` | `cudaq.adjoint(K, *args)` at top level | Hand-roll inverse if `K` is used inside `cudaq.control` |
| `qc.control(n)` | `cudaq.control(K, controls, *args)` | `K` must not contain `cudaq.adjoint` |
| `qc.compose(other, qubits)` | Direct function call with qview slices or explicit qubits | |
| Qiskit `ParameterVector` / binds | Pass parameter values as kernel arguments | No symbolic-vs-bound distinction |

For asymmetric controlled rotations, keep Qiskit's control-target orientation
exactly. Swapping arguments gives a different unitary.

## Bit Ordering and Count Keys

CUDA-Q and Qiskit stringify measurement results differently. Keep ordering
changes at the port boundary: allocation convention, measurement return list,
or final count-key formatting.

| API | Convention |
|---|---|
| `cudaq.sample` with `mz(qview)` | Keys follow qview order; `qview[0]` is leftmost |
| Multiple CUDA-Q `qvector` allocations | Keys follow allocation order, not `mz` call order |
| `cudaq.sample(..., explicit_measurements=True)` | Keys follow measurement-call order |
| `cudaq.run` returning `List[bool]` | Join return-list element 0 as leftmost |
| `cudaq.get_state(K)[i]` | `format(i, f"0{N}b")` matches CUDA-Q sample key order for the same kernel |
| Qiskit `get_counts` | Classical bit `c[0]` is displayed rightmost |

Mapping rules:

- To match Qiskit keys like `{prefix}{c[N-1]}...{c[0]}`, build a CUDA-Q
  return list in that same displayed order.
- To match a Qiskit register without a prefix, allocate CUDA-Q qubits so
  `qubits[N - 1 - q]` corresponds to Qiskit's `qr[q]`.
- If a Qiskit oracle decomposes an integer and reverses its bit list with
  `[::-1]`, check whether the CUDA-Q port should remove that reverse because
  CUDA-Q `register[0]` appears leftmost.
- CUDA-Q does not add spaces between classical registers in count keys.

When keys are wrong but counts are otherwise plausible, fix ordering at the
boundary, not by changing the algorithm. Use deterministic 2- or 3-qubit probes
and compare exact raw keys before relying on aggregate fidelity.

## Floating-Point Precision

CUDA-Q and Qiskit commonly default to different statevector precision:

| Framework | Typical backend | Default amplitude precision | Bytes per complex amplitude |
|---|---|---:|---:|
| CUDA-Q | `nvidia` / cuStateVec | fp32 | 8 |
| Qiskit | Aer statevector/GPU | fp64 | 16 |

Implications:

- Deep circuits or many small rotations can diverge between fp32 and fp64.
- Transpiling through small bases can add rotations and increase rounding drift.
- For cross-framework fidelity comparisons, use matching precision unless the
  task is explicitly default-vs-default comparison.

To match precision, adjust one side or the other (not both):

```python
# Option A: raise CUDA-Q to fp64 to match Aer's default
cudaq.set_target("nvidia", option="fp64")

# Option B: lower Aer to fp32 to match CUDA-Q's default
sim = AerSimulator(device="GPU", precision="single")
```

CUDA-Q fp64 roughly doubles statevector memory, reducing maximum GPU width.

## Framework-Decoupling Patterns

### Recursive Constructor to Flat Emitter

CUDA-Q kernels cannot recurse. If a Qiskit implementation recursively builds
subcircuits, port the recursion to a pure-Python emitter that produces arrays
consumed by a non-recursive kernel.

```python
def ucr_sequence_pure(n, theta):
    ry_angles, ctrl_indices = [], []

    def recurse(qubit_idx_list, theta_slice):
        if len(qubit_idx_list) == 1:
            ry_angles.append(float(theta_slice[0]))
            ctrl_indices.append(qubit_idx_list[0])
            ry_angles.append(float(theta_slice[1]))
        else:
            half = len(theta_slice) // 2
            recurse(qubit_idx_list[1:], theta_slice[:half])
            ctrl_indices.append(qubit_idx_list[0])
            recurse(qubit_idx_list[1:], theta_slice[half:])

    recurse(list(range(n)), list(theta))
    ctrl_indices.append(0)
    return ry_angles, ctrl_indices
```

If source code uses `sympy.combinatorics.GrayCode`, replace it with
`gray(i) = i ^ (i >> 1)` when possible.

### Source-Framework Import Blocker

Use this verification pattern after extracting pure helpers:

```python
import sys, importlib.abc

class _BlockSourceFramework(importlib.abc.MetaPathFinder):
    BLOCKED = ("qiskit", "qiskit_aer", "qiskit_ibm_runtime", "sympy")
    def find_spec(self, name, path, target=None):
        for blocked in self.BLOCKED:
            if name == blocked or name.startswith(blocked + "."):
                raise ModuleNotFoundError(f"refusing to import {name}")
        return None

sys.meta_path.insert(0, _BlockSourceFramework())
```

Then import and run the CUDA-Q port at a small representative configuration.

### Gate-Recorder Class

For deeply nested Qiskit constructors with `.inverse()` and `.control(k)`, use
a Python-side recorder whose methods mirror the source constructors and append
parallel arrays for a CUDA-Q kernel walker.

Default op-kind menu for modular-arithmetic-style circuits:

```text
1 = h
2 = x
3 = cx
4 = swap.ctrl
5 = r1(angle, t)
6 = r1.ctrl(angle, c, t)
7 = r1.ctrl(angle, c1, c2, t)
8 = rz.ctrl(angle, c, t)
```

For inverse calls, build a forward sub-sequence into a temporary recorder, then
append records in reverse order while negating rotation angles. Self-inverse
gates such as H, X, CX, and CSWAP keep angle 0.

Native-gate dispatch avoids per-circuit transpile cost and precision loss from
unnecessary transpilation. When `u3` / controlled `u3` exists in the installed
CUDA-Q API, emit it directly; otherwise decompose through supported rotations,
`cx`, and `swap`.

## Port Validation Gate

For every port:

1. Run the source implementation for the same deterministic inputs.
2. Run the CUDA-Q implementation with the same inputs.
3. Compare raw count keys, not only fidelity.
4. Test at least one small, nominal, and larger configuration when feasible.
5. Use stochastic tolerance only when shot noise is expected to dominate.
6. Re-run every previously failing configuration after changes.

When installed CUDA-Q behavior differs from the latest documentation while
debugging, review relevant documentation or source changes between the
installed version and latest before changing the port.

When stuck, copy subcircuit conventions from the exact source module being
ported. IQFT direction, qubit order, and gate flavor can differ across
implementations with similar names.

## External References

- CUDA-Q latest documentation: <https://nvidia.github.io/cuda-quantum/latest/>
- CUDA-Q examples in the local/source tree: `docs/sphinx/examples/python`
- CUDA-Q Academic: <https://github.com/NVIDIA/cuda-q-academic>
- CUDA-Q tests: `python/tests/kernel/test_kernel_features.py` and
  `python/tests/kernel/test_measure_handle.py` when present
- Companion skill: `cudaq-guide` (`/cudaq-guide author`) for execution API
  selection, kernel-language constraints, shared kernel patterns, resource
  metrics, and debugging workflow.
