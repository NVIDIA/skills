# Debugging Guide (Julia cuTile.jl)

---

## Julia-Specific Error Patterns

### Compilation Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `CodegenErrors: ... Unsupported function call during Tile IR compilation: isless with argument types (...Tile..., ...Tile...)` (stack shows `max`/`min`) | `max(a, b)` / `min(a, b)` on two tiles (non-broadcast) | Use `max.(a, b)` / `min.(a, b)` — broadcast dot syntax |
| `CodegenErrors: ... Unsupported function call during Tile IR compilation: <f> ...` | A Julia function with no Tile IR lowering is called on tiles | Use the broadcast form (`f.(tile)`) or a supported operation from `api-mapping.md` |
| `IRError` or `MethodError` mentioning `IRStructurizer` | Internal compiler bug | Do not work around — write a minimal reproducer and file upstream |
| `TypeError: in typeassert, expected Tile{...}, got Tile{...}` | Type mismatch in tile operation | Check `convert(ct.Tile{T}, tile)` calls |
| `MethodError` / `BoundsError` at launch | Wrong number of args in the `@cuda backend=cuTile` call | Verify arg count matches kernel signature exactly |
| `UndefVarError: cuTile not defined` at `@cuda backend=cuTile` | Only `import cuTile as ct` is in scope | Add `using cuTile: cuTile` |
| `CodegenErrors: ... UndefKeywordError: keyword argument dims not assigned` (stack shows `cat(::Tuple{...}, ::Int64)`) | Old `ct.cat((a, b), k)` call — removed in cuTile.jl 1.0, now falls through to `Base.cat` | Use `cat(a, b; dims=k)`, `vcat`/`hcat`, or `[a; b]` |
| `TileCompilerTimeoutError` | `tileiras` exceeded `compiler_timeout_seconds` | Raise the `compiler_timeout_seconds` preference, or simplify the kernel |
| `UndefVarError: X not defined` | Variable only defined in one `if` branch | Pre-define variable before the `if/else` |
| `UndefVarError: rsqrt not defined in Main` | Unqualified `rsqrt` — cuTile does not export it | Use `ct.rsqrt.(tile)` |

### Runtime Errors

| Error | Cause | Fix |
|-------|-------|-----|
| Wrong numerical results, correct shapes | Launch arg order doesn't match kernel signature | Args are positional — verify order |
| Wrong results after translating `ct.max(t, axis=…)` | Translated to `max` instead of `maximum` | `ct.max`/`ct.min` are reductions → `maximum`/`minimum`; Julia `max`/`min` are element-wise |
| Correct for first tile, wrong for subsequent tiles in loop | `ct.load` with `order` parameter has index positions not matching the remapped dimensions | **`order` remaps BOTH shape AND index** — see Critical Rule 16 |
| Wrong results at boundaries | Padding mode wrong or missing | Add `; padding_mode=ct.PaddingMode.Zero` |
| Off-by-one errors | 0-based index not converted to 1-based | Check `ct.bid`, `dims`, `ct.num_tiles` axis, `permutedims` axes |
| Silent wrong results | Data in the other layout than the kernel expects (Python `(M, N)` vs Julia `(N, M)`) | Reverse dims when converting; Julia `(N, M)` = Python row-major `(M, N)` |
| Correct but 3–8× slower than expected | Reduced/tiled axis is strided because Python's shape was kept (e.g. `(1, TILE)` loads on an `(M, N)` array) | Reverse dims so that axis is dim 1 (contiguous) — Critical Rule 7 |
| Kernel never finishes (GPU at 100%, `CUDA.synchronize()` hangs) | `occupancy=16` on a kernel with a tile reduction (e.g. `maximum(t; dims=1)`) hangs with cuTile.jl 1.0 / tileiras 13.3 on B200; plain load/store kernels are fine | Use `occupancy` ≤ 8 for such kernels |
| `CUDA error: illegal memory access` | Index out of bounds in gather/scatter | Check index computation and bounds |
| Kernel unexpectedly slow / recompiles | Array layout or `ct.Constant` value differs between launches (each is a separate specialization) | See cuTile.jl docs: *Compiling and Launching → What makes a distinct kernel* |

For common **test failure patterns** with symptoms and fixes, see [`testing.md`](testing.md) § Common Test Failure Patterns.

---

## Debug Commands

### Running Tests

```bash
# Run all Julia tests
julia --project=julia/ julia/test/runtests.jl

# Run a single test file
julia --project=julia/ julia/test/test_<op>.jl

# Dump emitted Tile IR bytecode (disassemble with `tileirdisasm <file>.cutile`)
JULIA_CUTILE_DUMP_BYTECODE=/tmp/julia_tiles julia --project=julia/ julia/test/test_<op>.jl
```

### Standalone Kernel Debugging

```bash
# Run a kernel file in isolation
julia --project=julia/ julia/kernels/<op>.jl
```

### Inspecting IR at a launch site

Wrap any launch with a `ct.@device_code_*` macro; argument types are derived from the actual `CuArray`s:

```julia
ct.@device_code_tiled      @cuda backend=cuTile blocks=grid kernel(args...)  # final Tile IR (MLIR text)
ct.@device_code_typed      @cuda backend=cuTile blocks=grid kernel(args...)  # typed Julia IR after overlays
ct.@device_code_structured @cuda backend=cuTile blocks=grid kernel(args...)  # after control-flow structurization

# Optimization remarks (tensor-core selection, alignment) need tileiras 13.4+,
# so they are not available with the default CUDA 13.3 toolchain
ct.@device_code_tiled remarks=true @cuda backend=cuTile blocks=grid kernel(args...)
```

Inside a kernel, `println(...)` and `ct.@assert cond "msg"` work for printf-style debugging.

### Compilation cache

The Tile IR → CUBIN step is cached on disk, content-addressed on the bytecode and compiler identity, so editing a kernel never produces a stale hit. To bypass it, set the `disk_cache = false` preference under `[cuTile]` in `LocalPreferences.toml`.
