---
name: cuequivariance
description: Use when defining cuEquivariance groups and irreps, building equivariant tensor products, or validating segmented polynomials with NumPy.
license: Apache-2.0
metadata:
  author: Mario Geiger <mgeiger@nvidia.com>
  tags: [equivariance, group-theory, tensor-products]
---

# cuequivariance: Groups, Irreps, and Segmented Polynomials

## Purpose

`cuequivariance` (imported as `cue`) provides two core abstractions:

1. **Group theory**: `Irrep` subclasses define irreducible representations of Lie groups (SO3, O3, SU2, or custom). `Irreps` manages collections with multiplicities.
2. **Segmented polynomials**: `SegmentedTensorProduct` describes tensor contractions over segments of varying shape, linked by `Path` objects carrying Clebsch-Gordan coefficients. `SegmentedPolynomial` wraps multiple STPs into a polynomial with named inputs/outputs. Two higher-level wrappers attach group representations:
   - `EquivariantPolynomial` — dense operands with `IrrepsAndLayout` metadata
   - `IrDictPolynomial` — operands already split by irrep, with per-group `Irreps` metadata for the `dict[Irrep, Array]` workflow

## Inputs

Required inputs: requested operation, group, input irreps and multiplicities, output irrep types, layout, batch shapes, dtype, and weight ownership. Take explicit user requirements first, then existing project definitions; state defaults for unspecified values. Derive operand sizes and output block order from the descriptor. For custom groups, also establish the selection rules, generators, and Clebsch-Gordan coefficients. Optional inputs include test rotations and a framework backend.

## Prerequisites

Use Python 3.10+ with `cuequivariance` and its NumPy, SciPy, opt_einsum, SymPy, and NetworkX dependencies. Reuse the project's interpreter and lockfile. If setup is needed outside an existing project, the bundled [CPU requirements](scripts/requirements.txt) pin the source revision used by these examples; install them with that interpreter's pip. Core NumPy execution requires no GPU stack or API credentials. Check the installed version and API availability before using `IrDictPolynomial` with an older package.

## Instructions

1. Check the installed version and requested backend. Install `cuequivariance` for CPU NumPy work; JAX and PyTorch execution use the separate `cuequivariance-jax` and `cuequivariance-torch` skills. Inspect available APIs before using newer features such as `IrDictPolynomial` in an older environment.
2. Establish the group, input irreps, multiplicities, layout, and batch shapes before building a descriptor. Preserve the descriptor's actual output block order; a regrouped irrep summary does not reorder its output array.
3. Prefer a built-in descriptor. For a custom group, validate parsing, group relations, selection rules, and Clebsch-Gordan covariance on CPU first. Complex mathematical representations do not imply complex CUDA support.
4. Execute a small deterministic NumPy example and check shapes and equivariance. `IrDictPolynomial` carries metadata; execute its `.polynomial` with operands split according to `.input_irreps` and `.output_irreps`, including weights first when present.
5. Convert `mul_ir` and `ir_mul` by transposing the multiplicity and representation axes **within each irrep block**. A flat reshape alone changes the represented features.

Source paths below are relative to the `cuequivariance/` project inside a source checkout. When using this skill separately, inspect the installed package with `cue.__file__` or consult https://github.com/NVIDIA/cuEquivariance. The repository is not required for the examples.

## Limitations

- Core descriptors and NumPy evaluation do not provide JAX or PyTorch execution kernels; use the corresponding frontend for framework execution and gradients.
- Custom-group mathematics must be checked independently. A valid complex representation does not establish support in a particular CUDA kernel.
- `IrDictPolynomial` describes split operands; it does not convert high-level dictionaries into the ordered arrays accepted by its underlying polynomial. Preserve the descriptor's metadata and layout when assembling those arrays.

## Troubleshooting

| Symptom | Check and correction |
| --- | --- |
| Import fails or an API is absent | Check the interpreter, installed version, and module location. Use the project's compatible dependencies or the bundled requirements when setting up a standalone environment. |
| Operand size or output shape is wrong | Compare each array with the descriptor's input/output dimensions; include the weights operand first when present. |
| Values disagree after a layout conversion | Transpose multiplicity and representation axes within each irrep block before flattening; keep output data in the descriptor's block order. |
| Custom-group equivariance fails | Check selection rules, generator relations, and CG covariance on a small NumPy example before changing execution backends. |

## Examples

The examples below cover custom groups, descriptors, layouts, and NumPy execution. Some snippets extend variables from the preceding snippet; the final complete example can be run independently.

## Defining a custom group

Subclass `cue.Irrep` (a frozen dataclass) and implement:

```python
from __future__ import annotations
import dataclasses
import re
from typing import Iterator
import numpy as np
import cuequivariance as cue

@dataclasses.dataclass(frozen=True)
class Z2(cue.Irrep):
    odd: bool  # dataclass field -- required for correct __eq__ and __hash__

    # No __init__ needed -- @dataclass(frozen=True) generates it: Z2(odd=True)

    @classmethod
    def regexp_pattern(cls) -> re.Pattern:
        # Pattern whose first group is passed to from_string
        return re.compile(r"(odd|even)")

    @classmethod
    def from_string(cls, string: str) -> Z2:
        if string not in ("odd", "even"):
            raise ValueError(f"Invalid Z2 irrep: {string!r}")
        return cls(odd=string == "odd")

    def __repr__(rep: Z2) -> str:
        return "odd" if rep.odd else "even"

    def __mul__(rep1: Z2, rep2: Z2) -> Iterator[Z2]:
        # Selection rule: which irreps appear in the tensor product rep1 x rep2
        return [Z2(odd=rep1.odd ^ rep2.odd)]

    @classmethod
    def clebsch_gordan(cls, rep1: Z2, rep2: Z2, rep3: Z2) -> np.ndarray:
        # Shape: (num_paths, rep1.dim, rep2.dim, rep3.dim)
        if rep3 in rep1 * rep2:
            return np.array([[[[1]]]])
        else:
            return np.zeros((0, 1, 1, 1))

    @property
    def dim(rep: Z2) -> int:
        return 1

    def __lt__(rep1: Z2, rep2: Z2) -> bool:
        # Ordering for sorting; dimension is compared first by the base class
        return rep1.odd < rep2.odd

    @classmethod
    def iterator(cls) -> Iterator[Z2]:
        # Must yield trivial irrep first
        for odd in [False, True]:
            yield Z2(odd=odd)

    def discrete_generators(rep: Z2) -> np.ndarray:
        # Shape: (num_generators, dim, dim)
        if rep.odd:
            return -np.ones((1, 1, 1))
        else:
            return np.ones((1, 1, 1))

    def continuous_generators(rep: Z2) -> np.ndarray:
        # Shape: (lie_dim, dim, dim) -- Z2 is discrete, so lie_dim=0
        return np.zeros((0, rep.dim, rep.dim))

    def algebra(self) -> np.ndarray:
        # Shape: (lie_dim, lie_dim, lie_dim) -- structure constants [X_i, X_j] = A_ijk X_k
        return np.zeros((0, 0, 0))


# Usage:
irreps = cue.Irreps(Z2, "3x odd + 2x even")  # dim=5
```

### Required methods summary

| Method | Returns | Purpose |
|--------|---------|---------|
| `regexp_pattern()` | `re.Pattern` | Parse string like `"1"`, `"0e"`, `"odd"` |
| `from_string(s)` | `Irrep` | Construct from matched string |
| `__repr__` | `str` | Canonical string form |
| `__mul__(a, b)` | `Iterator[Irrep]` | Selection rule for tensor product |
| `clebsch_gordan(a, b, c)` | `ndarray (n, d1, d2, d3)` | CG coefficients |
| `dim` (property) | `int` | Dimension of representation |
| `__lt__(a, b)` | `bool` | Ordering (dimension first, then custom) |
| `iterator()` | `Iterator[Irrep]` | All irreps, trivial first |
| `continuous_generators()` | `ndarray (lie_dim, dim, dim)` | Lie algebra generators |
| `discrete_generators()` | `ndarray (n, dim, dim)` | Finite symmetry generators |
| `algebra()` | `ndarray (lie_dim, lie_dim, lie_dim)` | Structure constants |

### Built-in groups

- **`cue.SO3(l)`**: 3D rotations. `l` is a non-negative integer. `dim = 2l+1`. String: `"0"`, `"1"`, `"2"`.
- **`cue.O3(l, p)`**: 3D rotations + parity. `p=+1` (even) or `p=-1` (odd). String: `"0e"`, `"1o"`, `"2e"`.
- **`cue.SU2(j)`**: Spin group. `j` is a non-negative half-integer. String: `"0"`, `"1/2"`, `"1"`.

## Irreps and layout

```python
irreps = cue.Irreps("SO3", "16x0 + 4x1 + 2x2")  # 16 scalars, 4 vectors, 2 rank-2
irreps.dim   # 16*1 + 4*3 + 2*5 = 38

for mul, ir in irreps:
    print(mul, ir, ir.dim)  # 16 0 1, then 4 1 3, then 2 2 5
```

`IrrepsLayout` controls memory order within each `(mul, ir)` block:

- `cue.ir_mul`: data ordered as `(ir.dim, mul)` — **used by all descriptors and ir_dict internally**
- `cue.mul_ir`: data ordered as `(mul, ir.dim)` — **used by nnx `dict[Irrep, Array]` and PyTorch**

`IrrepsAndLayout` combines irreps with a layout into a `Rep`:

```python
rep = cue.IrrepsAndLayout(cue.Irreps("SO3", "4x0 + 2x1"), cue.ir_mul)
rep.dim  # 10
```

## Building a SegmentedTensorProduct from scratch

The subscripts string uses Einstein notation. Operands are comma-separated, coefficient modes follow `+`.

```python
# Matrix-vector multiply: y_i = sum_j M_ij * x_j
d = cue.SegmentedTensorProduct.from_subscripts("ij,j,i")
d.add_segment(0, (3, 4))  # operand 0: matrix segment of shape (3, 4)
d.add_segment(1, (4,))     # operand 1: vector of size 4
d.add_segment(2, (3,))     # operand 2: output vector of size 3
d.add_path(0, 0, 0, c=1.0) # link segments 0,0,0 with coefficient=1.0

poly = cue.SegmentedPolynomial.eval_last_operand(d)  # last operand becomes output
[y] = poly(M_flat, x)  # numpy evaluation
```

### Multi-segment STP (how descriptors work internally)

Descriptors build STPs with multiple segments per operand. Each segment corresponds to an irrep block:

```python
# Linear equivariant map: output[iv] = sum_u weight[uv] * input[iu]
d = cue.SegmentedTensorProduct.from_subscripts("uv,iu,iv")

# Segment for l=1: ir_dim=3, mul_in=2, mul_out=5
s_in_0 = d.add_segment(1, (3, 2))    # input block
s_out_0 = d.add_segment(2, (3, 5))   # output block
d.add_path((2, 5), s_in_0, s_out_0, c=1.0)

# Segment for l=0: ir_dim=1, mul_in=4, mul_out=3
s_in_1 = d.add_segment(1, (1, 4))
s_out_1 = d.add_segment(2, (1, 3))
d.add_path((4, 3), s_in_1, s_out_1, c=1.0)
```

### Weights operand

For weighted tensor products (subscript starting with `uvw` or `uv`), the first operand is always weights. The weight segment shape is `(mul_1, mul_2, ...)` matching the multiplicity modes. The weights operand gets `new_scalars()` irreps since weights are invariant.

### CG coefficients as path coefficients

```python
d = cue.SegmentedTensorProduct.from_subscripts("uvw,iu,jv,kw+ijk")
# For each pair of input irreps and each output irrep in the selection rule:
for cg in cue.clebsch_gordan(ir1, ir2, ir3):
    # cg has shape (ir1.dim, ir2.dim, ir3.dim)
    d.add_path((mul1, mul2, mul3), seg_in1, seg_in2, seg_out, c=cg)
```

## Descriptors

All descriptors come in two variants:

- **Original** — returns `EquivariantPolynomial` with dense operands
- **`_ir_dict`** — returns `IrDictPolynomial` with operands already split by irrep

### EquivariantPolynomial descriptors

```python
# Fully connected tensor product (all input-output irrep combinations)
e = cue.descriptors.fully_connected_tensor_product(
    16 * cue.Irreps("SO3", "0 + 1 + 2"),
    16 * cue.Irreps("SO3", "0 + 1 + 2"),
    16 * cue.Irreps("SO3", "0 + 1 + 2"),
)

# Channelwise tensor product (same-channel only, sparse)
e = cue.descriptors.channelwise_tensor_product(
    64 * cue.Irreps("SO3", "0 + 1"), cue.Irreps("SO3", "0 + 1"),
    cue.Irreps("SO3", "0 + 1"), simplify_irreps3=True,
)

# Full (weightless) tensor product
e = cue.descriptors.full_tensor_product(
    cue.Irreps("SO3", "2x0 + 1x1"), cue.Irreps("SO3", "0 + 1"),
)

# Elementwise tensor product (paired channels)
e = cue.descriptors.elementwise_tensor_product(
    cue.Irreps("SO3", "4x0 + 4x1"), cue.Irreps("SO3", "4x0 + 4x1"),
)

# Linear equivariant map (weight x input)
e = cue.descriptors.linear(
    cue.Irreps("SO3", "4x0 + 2x1"),
    cue.Irreps("SO3", "3x0 + 5x1"),
)

# Spherical harmonics
e = cue.descriptors.spherical_harmonics(cue.SO3(1), [0, 1, 2, 3])

# Symmetric contraction (MACE-style)
e = cue.descriptors.symmetric_contraction(
    64 * cue.Irreps("SO3", "0 + 1 + 2"),
    64 * cue.Irreps("SO3", "0 + 1"),
    (1, 2, 3),
)
```

### IrDictPolynomial descriptors

Each `_ir_dict` constructor returns an `IrDictPolynomial`: `.polynomial` is a `SegmentedPolynomial` already split by irrep, and `.input_irreps` / `.output_irreps` describe the operand groups. Each `(mul, ir)` block in those groups corresponds to one polynomial operand of size `mul * ir.dim`.

`channelwise_tensor_product_ir_dict` always simplifies output irreps. To compare
with its dense counterpart, pass `simplify_irreps3=True` to
`channelwise_tensor_product`, or explicitly regroup the output **data** as well
as its irrep metadata. The third argument filters irrep types; it does not set
output multiplicities.

```python
# Channelwise tensor product
desc = cue.descriptors.channelwise_tensor_product_ir_dict(
    64 * cue.Irreps("SO3", "0 + 1"),
    cue.Irreps("SO3", "0 + 1"),
    cue.Irreps("SO3", "0 + 1"),
)
desc.polynomial       # SegmentedPolynomial
desc.input_irreps     # (weight_irreps, irreps1, irreps2)
desc.output_irreps    # (irreps_out,)

# Scale coefficients and inspect individual input operands
scaled_poly = desc.polynomial * 0.5
for i, op in enumerate(desc.polynomial.inputs):
    print(f"Input {i}: size={op.size}, num_segments={op.num_segments}")
```

Other constructors use the same operand metadata contract:

```python
# Fully connected tensor product
desc = cue.descriptors.fully_connected_tensor_product_ir_dict(irreps1, irreps2, irreps3)

# Full (weightless) tensor product
desc = cue.descriptors.full_tensor_product_ir_dict(irreps1, irreps2)

# Elementwise tensor product
desc = cue.descriptors.elementwise_tensor_product_ir_dict(irreps1, irreps2)

# Linear
desc = cue.descriptors.linear_ir_dict(irreps_in, irreps_out)

# Spherical harmonics
desc = cue.descriptors.spherical_harmonics_ir_dict(cue.O3(1, -1), [0, 1, 2, 3])

# Symmetric contraction
desc = cue.descriptors.symmetric_contraction_ir_dict(irreps_in, irreps_out, (1, 2, 3))
```

### split_polynomial_by_irreps

Use this helper to split an existing `SegmentedPolynomial` operand at irrep boundaries. Supply the operand's irreps explicitly because the bare polynomial has no representation metadata:

```python
poly = e.polynomial  # from an EquivariantPolynomial
poly = cue.split_polynomial_by_irreps(poly, 2, irreps_sh)   # split input 2
poly = cue.split_polynomial_by_irreps(poly, 1, irreps_in)   # split input 1
poly = cue.split_polynomial_by_irreps(poly, -1, irreps_out) # split output
```

### EquivariantPolynomial key methods

```python
e.inputs     # tuple of Rep (group representations for each input)
e.outputs    # tuple of Rep
e.polynomial # the underlying SegmentedPolynomial

# Numpy evaluation
[out] = e(weights, input1, input2)

# Preparing for uniform_1d execution (see cuequivariance_jax SKILL.md)
e_ready = e.squeeze_modes().flatten_coefficient_modes()

# Split dense operands using the representation metadata already attached to e
e_split = e.split_operand_by_irrep(1).split_operand_by_irrep(-1)

# Scale all coefficients
e_scaled = e * 0.5

# Fuse compatible STPs
e_fused = e.fuse_stps()
```

### normalize_paths_for_operand

Called internally by descriptors. Normalizes path coefficients so that a random input produces unit-variance output for the specified operand. Critical for numerical stability.

## SegmentedPolynomial structure

```python
poly = e.polynomial
poly.num_inputs    # number of input operands
poly.num_outputs   # number of output operands
poly.inputs        # tuple of SegmentedOperand
poly.outputs       # tuple of SegmentedOperand
poly.operations    # tuple of (Operation, SegmentedTensorProduct)

# Each operation maps buffers to STP operands
for op, stp in poly.operations:
    print(op.buffers)  # e.g., (0, 1, 2) means inputs[0], inputs[1] -> outputs[0]
    print(stp.subscripts)
```

### SegmentedOperand

```python
operand = poly.inputs[0]
operand.num_segments     # how many segments
operand.segments         # tuple of shape tuples, e.g., ((3, 4), (1, 2))
operand.size             # total flattened size (sum of products of segment shapes)
operand.ndim             # number of dimensions per segment
operand.all_same_segment_shape()  # True if all segments have identical shape
operand.segment_shape    # the common shape (only if all_same_segment_shape)
```

## Custom equivariant polynomial from scratch

```python
import numpy as np
import cuequivariance as cue

# Build a fully-connected SO3(1)xSO3(1)->SO3(0) tensor product manually
cg = cue.clebsch_gordan(cue.SO3(1), cue.SO3(1), cue.SO3(0))  # shape (1, 3, 3, 1)

d = cue.SegmentedTensorProduct.from_subscripts("uvw,iu,jv,kw+ijk")
d.add_segment(1, (3, 4))   # input1: 4x SO3(1), shape=(ir_dim, mul)
d.add_segment(2, (3, 4))   # input2: 4x SO3(1)
d.add_segment(3, (1, 16))  # output: 16x SO3(0) (4*4 fully connected)

for c in cg:
    d.add_path((4, 4, 16), 0, 0, 0, c=c)

d = d.normalize_paths_for_operand(-1)

poly = cue.SegmentedPolynomial.eval_last_operand(d)
ep = cue.EquivariantPolynomial(
    [
        cue.IrrepsAndLayout(cue.Irreps("SO3", "4x1").new_scalars(d.operands[0].size), cue.ir_mul),
        cue.IrrepsAndLayout(cue.Irreps("SO3", "4x1"), cue.ir_mul),
        cue.IrrepsAndLayout(cue.Irreps("SO3", "4x1"), cue.ir_mul),
    ],
    [cue.IrrepsAndLayout(cue.Irreps("SO3", "16x0"), cue.ir_mul)],
    poly,
)

# Numpy evaluation
w = np.random.randn(ep.inputs[0].dim)
x = np.random.randn(ep.inputs[1].dim)
y = np.random.randn(ep.inputs[2].dim)
[out] = ep(w, x, y)
```

## Key file locations

| Component | Path |
|-----------|------|
| `Irrep` base class | `cuequivariance/group_theory/representations/irrep.py` |
| `Rep` base class | `cuequivariance/group_theory/representations/rep.py` |
| `SO3` | `cuequivariance/group_theory/representations/irrep_so3.py` |
| `O3` | `cuequivariance/group_theory/representations/irrep_o3.py` |
| `SU2` | `cuequivariance/group_theory/representations/irrep_su2.py` |
| `Irreps` | `cuequivariance/group_theory/irreps_array/irreps.py` |
| `IrrepsLayout` | `cuequivariance/group_theory/irreps_array/irreps_layout.py` |
| `IrrepsAndLayout` | `cuequivariance/group_theory/irreps_array/irreps_and_layout.py` |
| `SegmentedTensorProduct` | `cuequivariance/segmented_polynomials/segmented_tensor_product.py` |
| `SegmentedPolynomial` | `cuequivariance/segmented_polynomials/segmented_polynomial.py` |
| `EquivariantPolynomial` | `cuequivariance/group_theory/equivariant_polynomial.py` |
| `IrDictPolynomial` | `cuequivariance/group_theory/ir_dict_polynomial.py` |
| Descriptors | `cuequivariance/group_theory/descriptors/` |
| Tensor product descriptors | `cuequivariance/group_theory/descriptors/irreps_tp.py` |
| `spherical_harmonics` | `cuequivariance/group_theory/descriptors/spherical_harmonics_.py` |
| `symmetric_contraction` | `cuequivariance/group_theory/descriptors/symmetric_contractions.py` |
