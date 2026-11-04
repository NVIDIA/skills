# Testing & Verification Guide (Julia cuTile.jl)

Julia kernels are tested using Julia's **native `Test` stdlib** — NOT through Python/pytest.
Tests live in `julia/test/` and run directly with `julia`.

---

## Architecture: How Julia Tests Work

```
julia --project=julia/ julia/test/runtests.jl
  ↓
julia/test/runtests.jl              # Test runner (@testset, includes test files)
  ↓
julia/test/test_<op>.jl             # Per-op test file
  ↓ include()
julia/kernels/<op>.jl               # Julia kernel (cuTile.jl) — bridge functions
  ↓
Host function launches the kernel (@cuda backend=cuTile ...) asynchronously; Array(result) waits for it
  ↓
Compare result vs reference (NNlib.jl, manual CPU computation, or CUDA.jl builtins)
```

---

## Test Command Reference

```bash
# Run all Julia tests
julia --project=julia/ julia/test/runtests.jl

# Run a single test file directly
julia --project=julia/ julia/test/test_softmax.jl

# Dump Tile IR bytecode for compilation issues
JULIA_CUTILE_DUMP_BYTECODE=/tmp/julia_tiles julia --project=julia/ julia/test/test_<op>.jl
```

---

## Writing a New Julia Test

### Step 1: Create test file `julia/test/test_<op>.jl`

```julia
using Test
using CUDA

# Load kernel
const KERNEL_DIR = joinpath(@__DIR__, "..", "kernels")
include(joinpath(KERNEL_DIR, "<op>.jl"))

@testset "<Op> Kernel" begin
    @testset "basic correctness" begin
        M, N = 128, 256
        x_gpu = CUDA.rand(Float32, M, N)
        out_gpu = similar(x_gpu)

        my_op!(out_gpu, x_gpu)

        expected = reference_impl(Array(x_gpu))
        @test Array(out_gpu) ≈ expected atol=1e-5
    end
end
```

### Step 2: Register in `julia/test/runtests.jl`

```julia
@testset "TileGym Julia Kernels" begin
    include(joinpath(TEST_DIR, "test_add.jl"))
    include(joinpath(TEST_DIR, "test_matmul.jl"))
    include(joinpath(TEST_DIR, "test_softmax.jl"))
    include(joinpath(TEST_DIR, "test_<op>.jl"))     # ← ADD THIS
end
```

---

## Reference Implementations

Use these for ground-truth comparison in tests:

| Operation | Reference | Package |
|-----------|-----------|---------|
| softmax | `NNlib.softmax(x; dims=1)` (per column of an `(N, M)` matrix — Python's per-row on `(M, N)`) | NNlib.jl |
| matmul | `A * B` (BLAS) | stdlib |
| batched matmul | `NNlib.batched_mul(A, B)` | NNlib.jl |
| attention | `NNlib.dot_product_attention(q, k, v; nheads=H)` | NNlib.jl |
| relu / gelu / silu | `NNlib.relu(x)` / `NNlib.gelu(x)` / `NNlib.swish(x)` | NNlib.jl |
| layer_norm | manual: `(x .- mean) ./ sqrt.(var .+ eps)` | manual |
| rms_norm | manual: `x ./ sqrt.(mean(x.^2) .+ eps)` | manual |
| add | `x .+ y .* alpha` | stdlib |

For simple ops (add, transpose), a manual CPU reference is fine.
For complex ops (attention, softmax), prefer NNlib.jl.

---

## Numerical Tolerances

| Precision | rtol | atol | Notes |
|-----------|------|------|-------|
| Float32 | 1e-3 | 1e-3 | Standard precision |
| Float32 + TF32 matmul | 1e-2 | 1e-1 | TF32 tensor cores have ~10-bit mantissa |
| Float16 | 1e-2 | 1e-2 | Half precision (if supported) |
| BFloat16 | 1e-2 | 1e-2 | Brain float (if supported) |
| Int32/64 | 0 | 0 | Exact match |

**Relax to 2x** for: reductions, transcendentals (`exp`, `log`, `sqrt`), chained ops, large tensors.

---

## Common Test Failure Patterns

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CodegenErrors: ... Unsupported function call ...: isless` (stack shows `max`) | `max(a, b)` on tiles | Use `max.(a, b)` (broadcast dot) |
| `IRError` or `MethodError` mentioning `IRStructurizer` | Internal compiler bug | Do not work around — file upstream with minimal reproducer |
| All zeros in output | Launch arg order wrong | Verify args map positionally to kernel params |
| Slight numerical drift | Reduction order differs | Increase tolerance to 2x default |
| Transposed results | Python `(M, N)` data fed to a Julia `(N, M)` kernel (or vice versa) | Reverse the dims when moving data between layouts: Julia `(N, M)` = Python row-major `(M, N)` |
| Correct but 3–8× slower than expected | Python shape kept, so the reduced/tiled axis is strided (e.g. `dims=2` on `(M, N)`) | Reverse dims so the axis is dim 1 — see `api-mapping.md` § Memory Layout Considerations |
| `UndefVarError: rsqrt not defined` | Unqualified `rsqrt` (not exported by cuTile) | Use `ct.rsqrt.(tile)` |

---

## Verification Checklist

Before marking a Julia kernel conversion complete:

- [ ] `julia --project=julia/ julia/test/runtests.jl` passes
- [ ] `validate_cutile_jl.py` passes on the `.jl` kernel file (no longer flags `for` loops)
- [ ] No NaN/Inf in output
- [ ] Tested at least one non-power-of-2 shape
- [ ] Tested at least one non-tile-aligned dimension
