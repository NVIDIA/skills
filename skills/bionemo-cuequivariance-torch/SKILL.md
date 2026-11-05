---
name: cuequivariance-torch
description: Use when building or debugging cuEquivariance PyTorch tensor products or equivariant GNNs, especially CPU references, layouts, indexing, and gradients.
license: Apache-2.0
metadata:
  author: Mario Geiger <mgeiger@nvidia.com>
  tags: [pytorch, equivariance, tensor-products]
---

# cuEquivariance in PyTorch

## Purpose

Execute `cuequivariance` (`cue`) descriptors with `cuequivariance_torch` (`cuet`): CPU references, CUDA backends, equivariant layers, and autograd. Preserve the project's environment.

## Inputs

Required inputs: operation, SO3/O3 irreps, layout, device, dtype, and weight ownership; infer from the request or code. Graphs also need source/destination counts and indices. Optional inputs: backend, performance, export targets.

## Instructions

1. Probe once and reuse working PyTorch. **Never recursively delete directories during setup or recovery**, including partial temporary environments; use a fresh unique path when isolation is needed.
2. Set `method="naive"` at construction for CPU. Match parameters, features, weights, and references in device and dtype. Use float64 for small numerical and gradient checks.
3. Specify layout and weight ownership explicitly. Derive sizes and output order from the module/descriptor; do not guess multiplicities from a filter.
4. Finish installation before running the saved script with the completion recipe below. A session/task ID, empty output, or a file write is not completion. Poll that session until it exits and inspect its results.
5. For an API mismatch, inspect the installed signature/source once. Adapt or explain the requirement; preserve the requested library.

## Prerequisites and environment reuse

Probe imports independently with the project's Python; a missing frontend must not hide working PyTorch:

```python
import importlib
import sys

print("python:", sys.executable, sys.version.split()[0])
available, import_errors = {}, {}
for name in ("torch", "cuequivariance", "cuequivariance_torch"):
    try:
        module = importlib.import_module(name)
        available[name] = module
        print(name, getattr(module, "__version__", "unknown"), module.__file__)
    except (ImportError, OSError) as error:
        import_errors[name] = f"{type(error).__name__}: {error}"
        print(name, import_errors[name])
if "torch" in available:
    torch = available["torch"]
    print("CUDA build:", torch.version.cuda, "available:", torch.cuda.is_available())
```

Choose the next step from the probe:

- Imports work: retain the interpreter; install nothing.
- Only cuEquivariance is missing: install `cuequivariance-torch` in that interpreter; it does not require replacing PyTorch.
- PyTorch is missing: check the project interpreter (e.g. `.venv/bin/python`). If the requested installation is inaccessible, say so; a new environment cannot verify it.
- Binary/shared-library mismatch: diagnose the reported package/path and compatibility.

Use one interpreter for pip and execution. Honor project requirements and compatible core/frontend revisions. Retaining installed PyTorch takes priority over setup examples. Optional CUDA/optimization warnings are not failures.

For isolation, create a **new unique** environment. From this skill's directory, use host pip 22.3+ and the bundled [hashed CPU wheels](scripts/torch-cpu-wheels.txt), then [requirements](scripts/requirements.txt):

```bash
set -e
cueq_python=python3  # Set to the selected project interpreter's path.
cueq_env=$(mktemp -d "${TMPDIR:-/tmp}/cueq-cpu.XXXXXX")
"$cueq_python" -m venv --without-pip "$cueq_env"
python3 -m pip --python "$cueq_env" install --cert "${PIP_CERT:-/etc/ssl/certs/ca-certificates.crt}" -q --no-deps -r scripts/torch-cpu-wheels.txt
python3 -m pip --python "$cueq_env" install --cert "${PIP_CERT:-/etc/ssl/certs/ca-certificates.crt}" -q -r scripts/requirements.txt
cueq_python="$cueq_env/bin/python"
"$cueq_python" -c 'import sys, torch, cuequivariance_torch; assert torch.version.cuda is None; print("CPU setup complete:", sys.executable, torch.__version__, cuequivariance_torch.__version__)'
```

Retain `cueq_python` for execution. For other platforms, use project setup. Without host pip, use installed `uv` with these requirements.

Use approved `PIP_CERT` for a private CA; retain TLS verification. If downloading fails, try a configured CPU mirror/cache or report the blocker. Do not replace CPU wheels with CUDA builds. Finish installs before retrying. See [pip's certificates](https://pip.pypa.io/en/stable/topics/https-certificates/).

### Complete and inspect a run

Save the preamble and checks in `check.py`. Set `cueq_python` to the probe's `sys.executable`, or retain it from setup. Run **after setup exits successfully**:

```bash
set -e
cueq_run=$(mktemp -d "${TMPDIR:-/tmp}/cueq-check.XXXXXX")
printf 'run directory: %s\n' "$cueq_run"
if "$cueq_python" -u check.py >"$cueq_run/output.txt" 2>&1; then
    cueq_status=0
else
    cueq_status=$?
fi
printf '%s\n' "$cueq_status" >"$cueq_run/exit-code.txt"
tail -n 30 "$cueq_run/output.txt"
printf 'exit code: %s\n' "$cueq_status"
test "$cueq_status" -eq 0
```

The exit-code file appears only when Python finishes. If the tool yields, poll its session, then inspect the file and numerical output. Deliver the script, command, observed results, and backend explanation; label unrun checks. Keep install output separate.

## Limitations and backends

| Method | Use and constraints |
| --- | --- |
| `naive` | Pure PyTorch reference implementation; supported CPU path and autograd. |
| `uniform_1d` | CUDA; uniform segment shapes within each operand and a single mode after descriptor transformations. |
| `fused_tp` | CUDA contractions with 3 or 4 operands; math dtype float32/float64. Some versions fall back on CPU, but select `naive` explicitly for a CPU reference. |
| `indexed_linear` | CUDA indexed linear operations; indices must be sorted. |

High-level defaults vary: `SphericalHarmonics` selects `uniform_1d`, `FullyConnectedTensorProduct` selects `fused_tp`, `Linear` selects `naive`, and `ChannelWiseTensorProduct`/`SymmetricContraction` select according to descriptor compatibility. Moving tensors to CPU does not change every constructor's backend choice. CUDA acceleration requires compatible PyTorch, GPU, and `cuequivariance_ops_torch` builds; CPU checks do not establish GPU performance or export compatibility.

## Layouts and weights

- `cue.mul_ir` is the high-level default, compatible with e3nn: each block is `(multiplicity, irrep_dimension)`. Core descriptors commonly use `cue.ir_mul`: `(irrep_dimension, multiplicity)`. Check each descriptor operand's `.layout`.
- Convert **each irrep block**, not the whole flattened tensor with a reshape. `cuet.TransposeIrrepsLayout(irreps, source=..., target=..., use_fallback=True)` performs the blockwise conversion using PyTorch.
- For externally supplied weights set `internal_weights=False`. Shared weights have shape `(1, module.weight_numel)`; distinct weights per item have shape `(batch, module.weight_numel)` and require `shared_weights=False`. A gathered graph operation uses the **edge count** for this batch dimension.
- `filter_irreps_out` selects allowed irrep types, not a requested output representation with prescribed multiplicities. Read `module.irreps_out`; preserve its actual order when constructing the next layer or reference.
- For training, create learned parameters or an edge-weight network in `__init__`. Do not draw new random weights inside `forward()`.

## Examples

Each CPU example runs independently with this preamble; preserve layout, indexing, and gradient contracts when adapting inputs.

```python
import numpy as np
import torch
import cuequivariance as cue
import cuequivariance_torch as cuet

torch.manual_seed(0)
dtype = torch.float64
print("torch:", torch.__version__, "CUDA build:", torch.version.cuda,
      "CUDA available:", torch.cuda.is_available(), "device: cpu", "dtype:", dtype,
      "method: naive")
```

### CPU tensor product and backward

Use `naive` for **both** CPU modules. Here weights are internal trainable parameters.

```python
irreps = cue.Irreps("SO3", "0 + 1")
sh = cuet.SphericalHarmonics([0, 1], normalize=False, method="naive")
tp = cuet.FullyConnectedTensorProduct(
    irreps, irreps, irreps, layout=cue.mul_ir, method="naive", dtype=dtype,
)
features = torch.randn(2, irreps.dim, dtype=dtype, requires_grad=True)
vectors = torch.tensor([[0.3, -0.7, 1.1], [1.2, 0.4, -0.2]],
                       dtype=dtype, requires_grad=True)
result = tp(features, sh(vectors))
assert result.device.type == "cpu" and result.shape == (2, irreps.dim)
result.square().sum().backward()
gradient_norms = {}
for name, tensor in [("features", features), ("vectors", vectors), *tp.named_parameters()]:
    assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    assert tensor.grad.abs().sum() > 0
    gradient_norms[name] = tensor.grad.norm().item()
print("CPU tensor product:", tuple(result.shape), "gradient norms:", gradient_norms)
```

### Spherical harmonics and a NumPy reference

`normalize=False` evaluates the raw polynomial; `normalize=True` first normalizes the input vectors. Match that choice in the reference and use nonzero vectors for gradient checks.

```python
ls = [0, 1, 2]
sh = cuet.SphericalHarmonics(ls, normalize=False, method="naive")
vectors = torch.tensor([[0.3, -0.7, 1.1], [1.2, 0.4, -0.2]],
                       dtype=dtype, requires_grad=True)
values = sh(vectors)
descriptor = cue.descriptors.spherical_harmonics(cue.SO3(1), ls)
reference = descriptor(vectors.detach().numpy())[0]
np.testing.assert_allclose(values.detach().numpy(), reference, atol=1e-12)
assert values.shape == (2, sum(2 * l + 1 for l in ls))
gradcheck_ok = torch.autograd.gradcheck(sh, (vectors,))
assert gradcheck_ok
print("Spherical harmonics:", tuple(values.shape), "normalize: False",
      "max error:", np.max(np.abs(values.detach().numpy() - reference)),
      "gradcheck:", gradcheck_ok)
```

### Linear: layout conversion and rotations

Use identical explicit weights for a layout comparison. For row-vector features, apply the representation matrix as `x @ D.T`.

```python
irreps_in = cue.Irreps("SO3", "3x0 + 2x1")
irreps_out = cue.Irreps("SO3", "2x0 + 4x1")
options = dict(internal_weights=False, shared_weights=True, method="naive", dtype=dtype)
linear = cuet.Linear(irreps_in, irreps_out, layout=cue.mul_ir, **options)
linear_ir = cuet.Linear(irreps_in, irreps_out, layout=cue.ir_mul, **options)
to_ir = cuet.TransposeIrrepsLayout(
    irreps_in, source=cue.mul_ir, target=cue.ir_mul, use_fallback=True
)
to_mul = cuet.TransposeIrrepsLayout(
    irreps_out, source=cue.ir_mul, target=cue.mul_ir, use_fallback=True
)
x = torch.randn(2, irreps_in.dim, dtype=dtype, requires_grad=True)
w = torch.randn(1, linear.weight_numel, dtype=dtype, requires_grad=True)
y = linear(x, w)
y_ir = to_mul(linear_ir(to_ir(x), w))
torch.testing.assert_close(y, y_ir)
rotation = np.array([0.2, -0.3, 0.1])
din, dout = [torch.from_numpy(cue.IrrepsAndLayout(r, cue.mul_ir).exp_map(rotation, []))
             for r in (irreps_in, irreps_out)]
rotated, expected_rotated = linear(x @ din.T, w), y @ dout.T
torch.testing.assert_close(rotated, expected_rotated)
grads = torch.autograd.grad(y.square().sum(), (x, w))
for g in grads:
    assert torch.isfinite(g).all() and g.abs().sum() > 0
print("Linear:", tuple(y.shape), "shared external weights:", tuple(w.shape),
      "layout error:", (y - y_ir).detach().abs().max().item(),
      "rotation error:", (rotated - expected_rotated).detach().abs().max().item(),
      "gradient norms:", [g.norm().item() for g in grads])
```

### Channelwise product with external per-example weights

The core descriptor takes `(weights, input1, input2)` and returns a list; the high-level module takes `(input1, input2, weight=...)` and returns a tensor. Matching `ir_mul` layouts avoids transposition. Report `tp.irreps_out` and preserve its actual order: the filter selects **types**, not multiplicities.

```python
a, b = cue.Irreps("SO3", "3x0 + 3x1"), cue.Irreps("SO3", "0 + 1")
keep = [cue.SO3(0), cue.SO3(1)]
tp = cuet.ChannelWiseTensorProduct(
    a, b, keep, layout=cue.ir_mul, shared_weights=False,
    internal_weights=False, method="naive", dtype=dtype,
)
x1, x2, w = [torch.randn(2, n, dtype=dtype, requires_grad=True)
             for n in (a.dim, b.dim, tp.weight_numel)]
y = tp(x1, x2, weight=w)
descriptor = cue.descriptors.channelwise_tensor_product(a, b, keep)
reference = descriptor(*(t.detach().numpy() for t in (w, x1, x2)))[0]
np.testing.assert_allclose(y.detach().numpy(), reference, atol=1e-12)
assert y.shape == (2, tp.irreps_out.dim)
grads = torch.autograd.grad(y.square().sum(), (x1, x2, w))
for g in grads:
    assert torch.isfinite(g).all() and g.abs().sum() > 0
print("Channelwise:", tuple(y.shape), "per-item weights:", tuple(w.shape),
      "filter selects types; actual ir_mul output order:", tp.irreps_out,
      "max error:", np.max(np.abs(y.detach().numpy() - reference)),
      "gradient norms:", [g.norm().item() for g in grads])
```

`cuet.FullyConnectedTensorProduct(a, b, irreps_out, method="naive", dtype=dtype)` accepts desired output irreps. Its weights are internal by default; use the same external-weight flags for caller-owned weights.

### SegmentedPolynomial: gather and scatter

Input/output index keys refer to **operand positions**, not feature channels. Supply `torch.int32` or `torch.int64` indices on the computation's device. `output_shapes` takes a two-dimensional **tensor**, whose first dimension determines the destination count; it does not take a shape tuple. Repeated receivers sum, and destinations without incoming edges remain zero.

```python
descriptor = cue.descriptors.linear(cue.Irreps("SO3", "3x0"), cue.Irreps("SO3", "2x0"))
sp = cuet.SegmentedPolynomial(descriptor.polynomial, method="naive")
nodes = torch.randn(5, 3, dtype=dtype, requires_grad=True)
w = torch.randn(1, descriptor.inputs[0].dim, dtype=dtype, requires_grad=True)
senders = torch.tensor([0, 2, 1, 0], dtype=torch.int64)
receivers = torch.tensor([1, 0, 1, 2], dtype=torch.int64)
[actual] = sp([w, nodes], input_indices={1: senders},
              output_indices={0: receivers}, output_shapes={0: nodes.new_empty(4, 1)})
[edges] = sp([w, nodes[senders]])
reference = nodes.new_zeros(4, descriptor.outputs[0].dim).index_add_(0, receivers, edges)
torch.testing.assert_close(actual, reference)
assert torch.count_nonzero(actual[3]) == 0
grads = torch.autograd.grad(actual.square().sum(), (nodes, w))
ref_grads = torch.autograd.grad(reference.square().sum(), (nodes, w))
for g, ref in zip(grads, ref_grads):
    torch.testing.assert_close(g, ref)
print("Gather/scatter:", tuple(actual.shape),
      "input/output index dtypes:", senders.dtype, receivers.dtype,
      "forward error:", (actual - reference).detach().abs().max().item(),
      "gradient errors:", [(g - r).abs().max().item() for g, r in zip(grads, ref_grads)],
      "isolated destination:", actual[3].detach().tolist())
```

Descriptor coefficients include normalization factors; retain them in references. High-level channelwise gathering/scattering uses `indices_1`, `indices_2`, `indices_out`, and `size_out` instead of dictionaries.

### Trainable message passing

This CPU reference uses a radial MLP to produce one weight vector per edge. Its parameters persist across calls, and `irreps_sh` matches the single copy of each spherical harmonic degree.

```python
class SimpleGNN(torch.nn.Module):
    def __init__(self, irreps_in, irreps_out):
        super().__init__()
        self.tp = cuet.ChannelWiseTensorProduct(
            irreps_in, cue.Irreps("SO3", "0 + 1 + 2"), layout=cue.mul_ir,
            shared_weights=False, internal_weights=False, method="naive",
        )
        self.radial = torch.nn.Sequential(
            torch.nn.Linear(1, 16), torch.nn.SiLU(),
            torch.nn.Linear(16, self.tp.weight_numel),
        )
        self.sh = cuet.SphericalHarmonics([0, 1, 2], normalize=True, method="naive")
        self.linear = cuet.Linear(
            self.tp.irreps_out, irreps_out, layout=cue.mul_ir, method="naive"
        )

    def forward(self, node_features, edge_vectors, senders, receivers, num_dst):
        weights = self.radial(edge_vectors.norm(dim=-1, keepdim=True))
        messages = self.tp(
            node_features, self.sh(edge_vectors), weight=weights,
            indices_1=senders, indices_out=receivers, size_out=num_dst,
        )
        return self.linear(messages)
```

Move the whole module and inputs to the same dtype/device (for example `model.double()` for float64 checks). Use nonzero edge vectors when normalizing spherical harmonics. After a scalar loss/backward pass, check gradients on the radial network as well as node features; repeated forwards should agree when no stochastic layers are present.

## Other operations and layers

Inspect the requested operation via `inspect.signature` and `inspect.getfile`. Key contracts:

| Operation | Contract |
| --- | --- |
| Species-indexed `Linear` | Set `weight_classes`; pass `weight_indices` of shape `(batch,)` and weights of shape `(weight_classes, weight_numel)`. |
| `SymmetricContraction` | MACE-style; uniform multiplicities, `contraction_degree`, `num_elements`, and integer element indices in `sc(x, indices)`. |
| `Rotation` / `Inversion` | `rot(gamma, beta, alpha, x)` uses YXY Euler angles; `inv(x)` uses O3 parity. Specify layout. Angle helpers: `encode_rotation_angle`, `vector_to_euler_angles`. |
| `layers.BatchNorm` | Equivariant normalization; configure irreps and layout. |
| `layers.FullyConnectedTensorProductConv` | Configure input/SH/output irreps and edge MLP; call with `(src_features, edge_sh, edge_emb, ((src, dst), (num_src, num_dst)))`. Supports `reduce="mean"` and optional `src_scalars`/`dst_scalars`. |

### Triangle operations and export

These paths require compatible optional GPU packages and their own shape checks. Do not install them for a CPU tensor-product request. Locate the installed `primitives/triangle.py` via `cuet.__file__`:

- `triangle_attention`: Q/K/V `(B,N,H,Q_or_K,D)`, bias `(B,1,H,Q,K)`. `mask_to_kv_lengths(prefix_mask)` supplies `kv_lengths`; arbitrary dense masks use `mask=...` and the fallback.
- `triangle_multiplicative_update`: input `(B,I,J,C)`, mask `(B,I,J)`, precision enum `cuet.TriMulPrecision`.
- `attention_pair_bias`: OpenFold3/Boltz contract, returns `(out, auxiliary)`; projection biases and query/key normalization keywords vary by version. Inspect its signature before assembling the weights.
- ONNX: pass `custom_translation_table=cuet.onnx_custom_translation_table()` to `torch.onnx.export`. TensorRT: `cuet.register_tensorrt_plugins()`. Validate the actual exported runtime.

## Troubleshooting

| Symptom | Check and recovery |
| --- | --- |
| Missing CUDA kernels on CPU | Pass `method="naive"` at module construction; retain the installed CPU PyTorch. |
| `opt_einsum_fx not available` warning | Optional optimization; continue the naive reference calculation without installing it. |
| Missing import after installation | Compare `sys.executable`, module paths, and the interpreter used with pip. Reuse the working interpreter. |
| Unexpected keyword or missing descriptor method | Inspect the installed signature/version. Use its documented equivalent or explain the mismatch; avoid repeated blind API guesses. |
| Correct shapes but wrong numerical values | Check per-block layout, output irrep order, normalization, and the shared/per-item weight convention. |
| Scatter shape/error or wrong destination sums | Use operand index keys, integral indices, a tensor in `output_shapes`, and accumulation for repeated receivers. |
| Missing/nonfinite gradients | Keep tensor operations in autograd; detach only the NumPy reference. Use nonzero vectors and a nonconstant loss; check every requested differentiable operand. |
