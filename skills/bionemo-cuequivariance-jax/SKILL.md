---
name: cuequivariance-jax
description: Use when writing or debugging cuEquivariance JAX polynomials, irrep dictionaries, Flax NNX layers, or CPU/CUDA execution and layouts.
license: Apache-2.0
metadata:
  author: Mario Geiger <mgeiger@nvidia.com>
  tags: [jax, equivariance, tensor-products]
---

# cuEquivariance in JAX

## Purpose

Execute `cuequivariance` (`cue`) descriptors with `cuequivariance_jax` (`cuex`). Preserve the project's framework and devices. Custom representations use core `cuequivariance`; PyTorch modules use `cuequivariance_torch`.

## Inputs

Required inputs: operation, SO3/O3 irreps, layout, batch shapes, dtype, device, and weight ownership. Derive them from the request/code; state defaults. Graphs also need indices and destination count. Operand sizes/output irreps come from descriptors. Optional inputs: backend/performance preferences.

## Instructions

1. Probe the project interpreter once and reuse working packages. **Environment setup/recovery must not recursively delete directories**, including temporary environments. Use a fresh unique path when isolation is needed.
2. Select `method="naive"` for CPU; Flax is needed only for NNX. High-level dictionary blocks are `(..., mul, ir.dim)`; uniform CUDA arrays differ.
3. Adapt one matching recipe into a script. Build descriptors outside transformed functions, broadcast leading input axes, and enable x64 **before array creation** for float64 checks.
4. Finish installation before running the saved script with the completion recipe below. A running session/task ID, empty output, or a file write is not a completed check. Poll the same session until it exits; then read its results before reporting success.
5. Investigate a concrete failure using its error and the affected installed API. Rerun affected checks after a fix. Once the requested checks pass, finish; avoid repeated probes, unrelated examples, or exploration of other backends for a CPU reference.

## Prerequisites and environment recovery

Probe imports independently so a missing frontend does not hide working JAX. For an NNX task, include `"flax.nnx"` in the same loop:

```python
import importlib
import sys

print("python:", sys.executable, sys.version.split()[0])
available, import_errors = {}, {}
for name in ("jax", "cuequivariance", "cuequivariance_jax"):
    try:
        module = importlib.import_module(name)
        available[name] = module
        print(name, getattr(module, "__version__", "unknown"), module.__file__)
    except (ImportError, OSError) as error:
        import_errors[name] = f"{type(error).__name__}: {error}"
print("import errors:", import_errors)
if "jax" in available:
    print("devices:", available["jax"].devices())
```

Reuse working packages; install only missing dependencies when needed. Honor the project's lockfile or supplied `scripts/requirements.txt` and compatible core/frontend revisions. JAX 0.11.1 / Flax 0.12.10 require **Python >=3.12**: select a compatible interpreter for those pins. The standalone combination below also supports Python 3.11; it does not replace project pins. Imports alone do not establish compatibility: cuEquivariance 0.13.0 with JAX 0.12.0 fails polynomial execution with a `segmented_polynomial_dce` argument-count error.

For isolation, keep the unique directory returned by `mktemp`; reuse it for reruns. This Linux recipe uses host pip 22.3+ and bypasses `ensurepip`:

```bash
set -e
cueq_python=python3  # Set to the selected project interpreter's path.
cueq_jax_env=$(mktemp -d "${TMPDIR:-/tmp}/cueq-jax.XXXXXX")
"$cueq_python" -m venv --without-pip "$cueq_jax_env"
python3 -m pip --python "$cueq_jax_env" install --cert "${PIP_CERT:-/etc/ssl/certs/ca-certificates.crt}" -q \
    "jax==0.10.2" "cuequivariance==0.13.0" "cuequivariance-jax==0.13.0"
cueq_python="$cueq_jax_env/bin/python"
JAX_PLATFORMS=cpu "$cueq_python" -c 'import jax, cuequivariance_jax; print("setup complete:", jax.__version__, cuequivariance_jax.__version__, jax.devices())'
```

Replace the package list with `-r` and the requirements path when prescribed; otherwise add `flax==0.12.7` to the same install command only for NNX. Keep this interpreter for execution; shell variables may need restoring in a new tool call. Without host pip, use an installed `uv`. The pip command uses the Linux system CA bundle; for a private CA use the environment's approved `PIP_CERT` bundle. Retain TLS verification. Diagnose the actual error before retrying; never reset directories or switch versions while an install is still running. If no compatible interpreter or trusted package source is available, report the blocker and unrun checks. See [pip's certificates](https://pip.pypa.io/en/stable/topics/https-certificates/).

### Complete and inspect a run

Save the preamble and relevant checks in `check.py`. Set `cueq_python` to the probe's `sys.executable`, or retain it from setup. Run **after setup exits successfully**; a fresh directory records output and status:

```bash
set -e
cueq_run=$(mktemp -d "${TMPDIR:-/tmp}/cueq-check.XXXXXX")
printf 'run directory: %s\n' "$cueq_run"
if JAX_PLATFORMS=cpu "$cueq_python" -u check.py >"$cueq_run/output.txt" 2>&1; then
    cueq_status=0
else
    cueq_status=$?
fi
printf '%s\n' "$cueq_status" >"$cueq_run/exit-code.txt"
tail -n 30 "$cueq_run/output.txt"
printf 'exit code: %s\n' "$cueq_status"
test "$cueq_status" -eq 0
```

If the tool yields, keep polling its session; the exit-code file is written only after Python finishes. Inspect that file and the numerical output. Empty output is not evidence that assertions ran. Deliver the script, actual command, and observed checks; label anything unrun. Keep setup output separate from the numerical results.

## Limitations and backend selection

| Route | Supported use |
| --- | --- |
| `segmented_polynomial(..., method="naive")` | Portable JAX execution, including CPU, JIT, gradients, and vmap. |
| `cuex.nnx.IrrepsLinear` | CPU/GPU channel mixing within matching irreps using JAX einsum. |
| `cuex.nnx.SphericalHarmonics` | CPU/GPU; normalizes vectors and uses the naive polynomial implementation. |
| `method="uniform_1d"` | CUDA; uniform segment shapes within each operand and a single mode after preparation. |
| `method="indexed_linear"`, `cuex.nnx.IrrepsIndexedLinear` | CUDA indexed linear route, using `cuex.Repeats`; not the ordinary CPU linear layer. |

CPU routes need no GPU dependencies; moving arrays to CPU does not change the method. CUDA requires compatible JAX/GPU/kernel builds. NNX spherical harmonics normalize vectors: use positive `eps` if zero vectors are possible.

## Examples

Use this preamble with one recipe and run with `JAX_PLATFORMS=cpu`. Nonconstant inputs expose layout errors; prints follow successful assertions.

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import cuequivariance as cue
import cuequivariance_jax as cuex
from cuequivariance_jax import ir_dict

dtype = jnp.float64
print("devices:", jax.devices())
```

### Spherical harmonics without Flax

This evaluates the **raw polynomial**; normalize vectors for unit-direction harmonics. O3 keys retain parity; SO3 keys do not.

```python
desc = cue.descriptors.spherical_harmonics_ir_dict(cue.O3(1, -1), [0, 1, 2])
vectors = jnp.array([[0.3, -0.7, 1.1], [1.2, 0.4, -0.2]], dtype=dtype)
values = cuex.segmented_polynomial(
    desc.polynomial, [vectors],
    [jax.ShapeDtypeStruct(vectors.shape[:-1] + (op.size,), dtype)
     for op in desc.polynomial.outputs],
    method="naive",
)
(irreps_out,) = desc.output_irreps
harmonics = {
    ir: value.reshape(vectors.shape[:-1] + (mul, ir.dim))
    for (mul, ir), value in zip(irreps_out, values)
}
for actual, expected in zip(values, desc.polynomial(np.asarray(vectors))):
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
assert {ir.l: v.shape for ir, v in harmonics.items()} == {
    0: (2, 1, 1), 1: (2, 1, 3), 2: (2, 1, 5),
}
print({str(ir): (v.shape, v.devices()) for ir, v in harmonics.items()})
```

### Dictionary layouts and exact round trips

`flat_to_dict(..., layout="ir_mul")` produces `(..., mul, ir.dim)` blocks. **`dict_to_flat` has no layout argument and emits `mul_ir`.** Return to `ir_mul` by transposing each block, not reshaping the whole array.

Explain the **unique-irrep key precondition**, even for regrouped inputs. Regroup duplicate blocks **with their data** using `RepArray.regroup()`; changing only metadata corrupts alignment. The round trip below recovers the regrouped buffer. Recovering an ungrouped order also needs its original block arrangement.

```python
def ir_mul_to_dict(irreps, flat):
    # Merge duplicate irrep keys and reorder their data together.
    data = cuex.RepArray(cue.IrrepsAndLayout(irreps, cue.ir_mul), flat).regroup()
    return data.irreps, ir_dict.flat_to_dict(data.irreps, data.array, layout="ir_mul")

def dict_to_ir_mul_flat(irreps, blocks):
    return jnp.concatenate([
        blocks[ir].swapaxes(-1, -2).reshape(
            blocks[ir].shape[:-2] + (mul * ir.dim,)
        )
        for mul, ir in irreps
    ], axis=-1)

irreps = cue.Irreps("SO3", "3x0 + 2x1").regroup()
flat_ir = jnp.arange(2 * irreps.dim).reshape(2, irreps.dim)
irreps, blocks = ir_mul_to_dict(irreps, flat_ir)
offset = 0
for mul, ir in irreps:
    size = mul * ir.dim
    expected = np.asarray(flat_ir[..., offset:offset + size]).reshape(2, ir.dim, mul)
    assert blocks[ir].shape == (2, mul, ir.dim)
    np.testing.assert_array_equal(blocks[ir], expected.swapaxes(-1, -2))
    offset += size
flat_mul = ir_dict.dict_to_flat(irreps, blocks)
restored_blocks = ir_dict.flat_to_dict(irreps, flat_mul, layout="mul_ir")
for ir in blocks:
    np.testing.assert_array_equal(restored_blocks[ir], blocks[ir])
np.testing.assert_array_equal(dict_to_ir_mul_flat(irreps, restored_blocks), flat_ir)
print("regrouped unique irrep keys; exact ir_mul/mul_ir round trip:",
      {str(ir): v.shape for ir, v in blocks.items()})
```

`ir_dict.assert_mul_ir_dict` checks block conventions. `mul_ir_dict(irreps, data)` broadcasts a template to keys without inferring feature dimensions.

### Dense polynomial: broadcasting, JIT, and a gradient

`segmented_polynomial` takes/returns lists. Channelwise products put weights first; shared weights need **no batch axis**. Broadcast leading input axes. `filter_irreps_out` selects types, not multiplicities; `e.inputs`/`e.outputs` carry dense operand layouts.

```python
e = cue.descriptors.channelwise_tensor_product(
    cue.Irreps("SO3", "3x0 + 3x1"), cue.Irreps("SO3", "0 + 1"),
    [cue.SO3(0), cue.SO3(1)],
)
poly = e.polynomial
rng = np.random.default_rng(7)
w = jnp.asarray(rng.normal(size=poly.inputs[0].size))
x = jnp.asarray(rng.normal(size=(3, 1, poly.inputs[1].size)))
y = jnp.asarray(rng.normal(size=(1, 2, poly.inputs[2].size)))

def run(weights, a, b):
    batch = np.broadcast_shapes(weights.shape[:-1], a.shape[:-1], b.shape[:-1])
    return cuex.segmented_polynomial(
        poly, [weights, a, b],
        [jax.ShapeDtypeStruct(batch + (op.size,), jnp.result_type(weights, a, b))
         for op in poly.outputs],
        method="naive",
    )[0]

reference = e(*(np.asarray(v) for v in (w, x, y)))[0]
eager, compiled = run(w, x, y), jax.jit(run)(w, x, y)
np.testing.assert_allclose(eager, reference, rtol=1e-12, atol=1e-12)
np.testing.assert_allclose(compiled, reference, rtol=1e-12, atol=1e-12)
assert reference.shape == (3, 2, poly.outputs[0].size)
gradient = jax.grad(lambda v: jnp.sum(run(v, x, y) ** 2))(w)
index = int(jnp.argmax(jnp.abs(gradient)))
assert abs(gradient[index]) > 1e-8
epsilon = 1e-5
loss_plus = np.sum(e(np.asarray(w.at[index].add(epsilon)), np.asarray(x), np.asarray(y))[0] ** 2)
loss_minus = np.sum(e(np.asarray(w.at[index].add(-epsilon)), np.asarray(x), np.asarray(y))[0] ** 2)
finite_difference = (loss_plus - loss_minus) / (2 * epsilon)
np.testing.assert_allclose(gradient[index], finite_difference, rtol=1e-6, atol=1e-7)
print("shape:", eager.shape, "eager/JIT max errors:",
      float(jnp.max(abs(eager - reference))), float(jnp.max(abs(compiled - reference))))
print("weight gradient / finite difference:", float(gradient[index]), finite_difference)
```

For gather/scatter, supply one `indices` entry per operand, inputs then outputs; `None` means unindexed. Output indices need a destination shape; repeated destinations accumulate. Verify against explicit edgewise summation.

### Split irrep polynomials on CPU

`channelwise_tensor_product_ir_dict` returns an `IrDictPolynomial`. Follow `input_irreps` order: weights, features, second input; each `(mul, ir)` is one operand. Follow `output_irreps` for outputs. These descriptors use `ir_mul`: transpose blocks before flattening; reverse on outputs. There is no `ir_dict.equivariant_polynomial` API.

```python
def split_naive(desc, groups):
    assert len(groups) == len(desc.input_irreps)
    operands = [
        group[ir].swapaxes(-1, -2).reshape(group[ir].shape[:-2] + (mul * ir.dim,))
        for irreps, group in zip(desc.input_irreps, groups)
        for mul, ir in irreps
    ]
    batch = np.broadcast_shapes(*(v.shape[:-1] for v in operands))
    outputs = cuex.segmented_polynomial(
        desc.polynomial, operands,
        [jax.ShapeDtypeStruct(batch + (op.size,), jnp.result_type(*operands))
         for op in desc.polynomial.outputs],
        method="naive",
    )
    values = iter(outputs)
    return tuple({
        ir: next(values).reshape(batch + (ir.dim, mul)).swapaxes(-1, -2)
        for mul, ir in irreps
    } for irreps in desc.output_irreps)

desc = cue.descriptors.channelwise_tensor_product_ir_dict(
    cue.Irreps("SO3", "3x0 + 3x1"), cue.Irreps("SO3", "0 + 1"),
    [cue.SO3(0), cue.SO3(1)],
)
rng = np.random.default_rng(8)
groups = [
    {ir: jnp.asarray(rng.normal(size=batch + (mul, ir.dim))) for mul, ir in irreps}
    for irreps, batch in zip(desc.input_irreps, [(), (3, 1), (1, 2)])
]
(result,) = split_naive(desc, groups)
numpy_inputs = [
    np.asarray(group[ir]).swapaxes(-1, -2).reshape(group[ir].shape[:-2] + (mul * ir.dim,))
    for irreps, group in zip(desc.input_irreps, groups) for mul, ir in irreps
]
expected = desc.polynomial(*numpy_inputs)
for (mul, ir), flat in zip(desc.output_irreps[0], expected):
    assert result[ir].shape == (3, 2, mul, ir.dim)
    reference = flat.reshape(3, 2, ir.dim, mul).swapaxes(-1, -2)
    np.testing.assert_allclose(result[ir], reference, rtol=1e-12, atol=1e-12)
    print("output:", str(ir), result[ir].shape, "max error:",
          float(jnp.max(abs(result[ir] - reference))))
```

### Flax NNX layers

`IrrepsLinear` requires regrouped irreps and `nnx.Rngs`. It mixes matching-irrep channels with `scale / sqrt(mul_in)` normalization; unmatched outputs are zero. Read `nnx.Param` weights with `[...]`. Row vectors rotate as `x @ D.T`.

```python
from flax import nnx
from cuequivariance_jax import nnx as cue_nnx

irreps_in = cue.Irreps("SO3", "3x0 + 2x1").regroup()
irreps_out = cue.Irreps("SO3", "2x0 + 4x1 + 2").regroup()
linear = cue_nnx.IrrepsLinear(irreps_in, irreps_out, scale=0.7, dtype=dtype, rngs=nnx.Rngs(9))
rng = np.random.default_rng(10)
features = {ir: jnp.asarray(rng.normal(size=(2, mul, ir.dim))) for mul, ir in irreps_in}
output = linear(features)
for mul, ir in irreps_out:
    assert output[ir].shape == (2, mul, ir.dim)
np.testing.assert_array_equal(output[cue.SO3(2)], 0)
for mul, ir in irreps_in:
    factor = linear.scale / np.sqrt(mul)
    expected = factor * np.einsum(
        "uv,...ui->...vi", np.asarray(linear.w[str(ir)][...]), features[ir])
    np.testing.assert_allclose(output[ir], expected, rtol=1e-12, atol=1e-12)
    print("irrep:", str(ir), "scale/sqrt(mul_in):", factor,
          "reference max error:", float(jnp.max(abs(output[ir] - expected))))
rotation = np.array([0.3, -0.2, 0.1])
rotated_output = linear({ir: v @ ir.exp_map(rotation, []).T for ir, v in features.items()})
for ir, value in output.items():
    expected = value @ ir.exp_map(rotation, []).T
    np.testing.assert_allclose(rotated_output[ir], expected, rtol=1e-12, atol=1e-12)
    print("output:", str(ir), value.shape, "rotation max error:",
          float(jnp.max(abs(rotated_output[ir] - expected))))
print("unmatched l=2 max magnitude:", float(jnp.max(abs(output[cue.SO3(2)]))))
```

`IrrepsNormalize(eps=..., scale=..., skip_scalars=...)` uses squared component norms averaged over multiplicities. `MLP(layer_sizes=..., activation=..., dtype=..., rngs=...)` uses scalar features and fan-in normalization. CUDA indexed-linear species counts must match grouped rows and sum to batch size.

### Dense RepArray interface

Use descriptor representation metadata. A single output returns one `RepArray`: `.array` holds data; `.reps` maps axes to representations.

```python
e = cue.descriptors.fully_connected_tensor_product(
    cue.Irreps("SO3", "2x0 + 1"), cue.Irreps("SO3", "0 + 1"),
    cue.Irreps("SO3", "0 + 2x1"),
)
keys = jax.random.split(jax.random.key(11), len(e.inputs))
inputs = [cuex.randn(key, rep, (2,), dtype) for key, rep in zip(keys, e.inputs)]
output = cuex.equivariant_polynomial(e, inputs, method="naive")
np.testing.assert_allclose(output.array, e(*(np.asarray(v.array) for v in inputs))[0],
                           rtol=1e-12, atol=1e-12)
print("RepArray shape:", output.shape, "devices:", output.array.devices())
```

## CUDA preparation and message passing

Prepare compatible dense descriptors with `e.squeeze_modes().flatten_coefficient_modes()` before `equivariant_polynomial(..., method="uniform_1d")`. Segment shapes must be uniform within each operand. Splitting by irrep still requires layout conversion.

`ir_dict.segmented_polynomial_uniform_1d(poly, inputs, outputs, input_indices=..., output_indices=...)` needs leaves shaped `(..., op.num_segments, *op.segment_shape)`, distinct from `(..., mul, ir.dim)`. For `_ir_dict` operands, transpose/flatten as in `split_naive`, then reshape using segment metadata. Match leaf order to operand order and output specs to segment suffixes. Reverse conversions before `IrrepsLinear`.

GNN weights belong to edges, features to nodes, harmonics to edge directions. Gather by senders, scatter by receivers, and size output by destination count. Initialize learned weights/edge networks once in the NNX initializer; split RNG keys for independent inputs. Source examples: `cuequivariance_jax/examples/nequip_nnx.py` and `mace_nnx.py`.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `ensurepip` unavailable | Use a new unique directory, `--without-pip`, and host pip `--python`; preserve partial environments. |
| Missing CUDA kernels on CPU | Select `naive` or ordinary NNX layers. |
| Float64 truncated / numerical mismatch | Check early x64, identical operands, dtype/normalization, layout, and operand order. |
| Dictionary mismatch | Regroup data with metadata; transpose per block. `dict_to_flat` emits `mul_ir`. |
| Missing output / bad broadcast | Specify every output; broadcast leading axes including weights. |
| Missing API / unexpected keyword | Inspect that installed symbol's signature/source and version. |

Locate source via `cuex.__file__`; relevant files are `ir_dict.py`, `nnx.py`, `equivariant_polynomial.py`, `rep_array/rep_array_.py`, and `segmented_polynomials/segmented_polynomial*.py`. API docs: https://docs.nvidia.com/cuda/cuequivariance/.
