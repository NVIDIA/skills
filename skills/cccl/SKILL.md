---
name: cccl
description: >-
  Use for CUDA C++ work. Use when asked to "write a kernel that ...".
  Use before hand-writing a CUDA kernel for something CCCL provides or can be composed of:
  sum/min/max/argmax reductions, prefix sums, sorting, top-k, histograms, stream compaction,
  per-segment or per-key reductions, search, merge, or an elementwise transform.
  Use for anything that touches NVIDIA CCCL (Thrust, CUB, libcu++ / cuda::std): writing, porting,
  reviewing, optimizing, debugging, fusing passes, streams, deterministic float results, cuda::std
  parallel algorithms, and the libcu++ runtime API (cuda::stream, cuda::launch). Use when Thrust/CUB
  code returns wrong results, is slow, or stops compiling or building after a CUDA 13 / CCCL 3.x upgrade.
  Not for Python (CuPy, PyTorch, cuda.compute) or for other CUDA libraries such as cuBLAS or cuDNN.
license: CC-BY-4.0
metadata:
  author: NVIDIA
  tags: [cuda, cccl, thrust, cub, libcudacxx]
---

# CCCL: Thrust, CUB, libcu++

CCCL (CUDA Core Compute Libraries) ships Thrust (STL-like parallel algorithms), CUB (device-, block-,
and warp-level primitives), and libcu++ (`cuda::std::`, `cuda::`). It is bundled with the CUDA Toolkit.

## Reach for CCCL before writing a kernel

A lot of kernels that people write by hand are already in CCCL, tuned per GPU and tested on the edge cases.
When asked for "a kernel that does X", check this table first, use the library call, and say so.

| Task | Call |
|---|---|
| Sum, min, max, any fold | `thrust::reduce` / `transform_reduce`, `cub::DeviceReduce` |
| Max or min with its position | `thrust::max_element` (first on ties), `cub::DeviceReduce::ArgMax(d_in, d_val, d_idx, n)` (`int64_t` index) |
| Prefix sum | `thrust::inclusive_scan` / `exclusive_scan`, `cub::DeviceScan` |
| Sort, top-k | `thrust::sort` / `sort_by_key`, `cub::DeviceRadixSort`, `cub::DeviceSegmentedSort`, `cub::DeviceTopK` |
| Histogram | `cub::DeviceHistogram::HistogramEven(d_img, d_hist, 257, 0, 256, n)` (256 bins = 257 levels) |
| Keep, drop, split | `thrust::copy_if` / `remove_if` / `partition`, `cub::DeviceSelect`, `cub::DevicePartition` |
| Count runs, run lengths | `thrust::unique_count`, `cub::DeviceRunLengthEncode` |
| Per-segment or per-key | `cub::DeviceSegmentedReduce`, `thrust::reduce_by_key`, `cub::DeviceSegmentedScan` |
| Search, first difference, merge | `thrust::find_if`, `thrust::mismatch`, `thrust::lower_bound`, `thrust::merge` |
| Elementwise map, index loop | `thrust::transform` / `for_each`, `cub::DeviceTransform`, `cub::DeviceFor::ForEachInExtents` |

## Workflow

1. Check the version first: CUDA 12 bundles CCCL 2.x, CUDA 13 bundles CCCL 3.x, which removed many
   APIs that older examples use. Headers are in `<ctk>/include/cccl/` (a symlink: use `grep -R`).
   To print the version (3005000 means 3.5.0):

   ```bash
   printf '#include <cuda/version>\nCCCL_VERSION\n' | nvcc -x cu -E - | tail -1
   ```
2. Build with `nvcc -O2 -std=c++17 -arch=native` (`--extended-lambda` for lambdas) and run it.
3. For performance work, time after a warm-up call and report measured before and after numbers.
4. Tell the user what was wrong and why, not only what changed.

## Fuse passes instead of chaining them

Try to fuse as much as possible. Each Thrust or CUB call is at least one kernel and one pass over
memory, and every intermediate `device_vector` is an allocation. Fold the elementwise steps into
the algorithm that consumes them with `make_zip_iterator`, `make_transform_iterator`,
`counting_iterator`, and `make_permutation_iterator`. For several results, reduce to a struct.

```cpp
struct Stats
{
  long long count;
  double sum;
  float max;
};

struct Energy
{
  __host__ __device__ Stats operator()(const thrust::tuple<float, float>& p) const
  {
    const float mass  = thrust::get<0>(p);
    const float speed = thrust::get<1>(p);
    const float e     = 0.5f * mass * speed * speed;
    return e > 300.0f ? Stats{1, e, e} : Stats{0, 0.0, 0.0f};
  }
};

struct Merge
{
  __host__ __device__ Stats operator()(const Stats& a, const Stats& b) const
  {
    return {a.count + b.count, a.sum + b.sum, a.max > b.max ? a.max : b.max};
  }
};

auto first = thrust::make_zip_iterator(mass.begin(), speed.begin());
Stats s    = thrust::transform_reduce(first, first + n, Energy{}, Stats{0, 0.0, 0.0f}, Merge{});
```

## Silent wrong results

- Thrust accumulates in the type of `init` (or the input type without one), not the output
  type. `reduce(floats, 0)` truncates to `int`; `reduce(ints)` overflows even into a `long long`;
  scans and sums over `uint8_t` or `bool` wrap. Pass an init of the type you want (`0.0f`, `0LL`).
- Reductions need a commutative operator. `thrust::reduce` and `cub::DeviceReduce` reorder
  operands, so composing matrices or affine maps gives a wrong answer with no error. Scans keep
  the order: use `thrust::inclusive_scan` with the same operator and take the last element.
- Raw device pointers need `thrust::device`. Without a policy Thrust runs on the host and segfaults.
- `reduce_by_key` and `unique` merge only consecutive equal keys (sort first). `HistogramEven`
  bins are `[lower, upper)`, so a sample equal to `upper` is dropped.
- `cub::DeviceReduce::Min` and `DeviceSegmentedReduce::Min` return `FLT_MAX`, not `+inf`, for
  empty or all-`+inf` input. Use `Reduce` with `cuda::minimum<>{}` and an explicit `+inf` init.
- `cub::DoubleBuffer` output is in `buf.Current()`, which may be either buffer. Radix sort with
  `begin_bit`/`end_bit` on signed or float keys comes out unsorted. Check every `cudaError_t`.

## Performance

- Never index a `device_vector` in a host loop: each `v[i]` is a `cudaMemcpy`. Copy once.
- One segmented call, not a loop of reductions. For uniform segments (matrix rows) use
  `cub::DeviceSegmentedReduce::Sum(d_in, d_out, num_segments, segment_size)`; for small variable
  ones `thrust::reduce_by_key` with a key per item; for large ones the offsets overload
  `Sum(d_in, d_out, num_segments, d_offsets, d_offsets + 1)`. The offsets overload uses a thread
  block per segment: 4M rows of 16 take 3.6 ms with it, 0.37 ms fixed-size. No custom kernel needed.
- Keep sorts on the radix path. A custom comparator or struct keys fall back to merge sort
  (2-6x slower). Sort plain keys with an index (`sort_by_key`), then `thrust::gather` the records.
  If keys fit in fewer bits, pass `begin_bit`/`end_bit` to `cub::DeviceRadixSort` (unsigned only).
- Thrust blocks by default. `thrust::device` and `thrust::cuda::par.on(s)` wait for the stream
  after every call, so streams never overlap. Use `thrust::cuda::par_nosync.on(s)`. Calls that return
  a value (`reduce`, `count`) still block: use `thrust::reduce_into(..., d_out, init)` or CUB instead.

## CUB execution environments

CCCL 3.x device algorithms have single-call overloads with no temp-storage arguments and an optional
environment last: a stream, or an `env{...}` of stream, memory pool, and requirements. They do not block.

```cpp
// A stream is an environment by itself
cub::DeviceReduce::Sum(d_in, d_out, n, cuda::stream_ref{stream});

// Stream plus a memory pool for the temporary storage
auto pool = cuda::device_default_memory_pool(cuda::devices[0]);
auto env  = cuda::std::execution::env{cuda::stream_ref{stream}, pool};
cub::DeviceSelect::If(d_in, d_out, d_count, n, pred, env);
```

## Determinism

Float results depend on the order of combination. Run to run on the same GPU:
- `cub::DeviceReduce`, `thrust::reduce`, `cub::DeviceSegmentedReduce`: reproducible (`run_to_run`).
- `cub::DeviceScan`, `thrust::inclusive_scan`: **not** reproducible by default (20 of 20 reruns differed).
- `thrust::reduce_by_key`, `inclusive_scan_by_key`: **not** reproducible for long key runs, and it
  cannot be requested. Use `cub::DeviceSegmentedReduce::Sum` over offsets taken from the sorted
  keys with a vectorized `thrust::lower_bound` (integer work is always reproducible).

Ask for a guarantee instead of rewriting the algorithm. It stays float and costs nothing:

```cpp
namespace ex = cuda::execution;

// Same bits on every run on this GPU
auto env = cuda::std::execution::env{cuda::stream_ref{stream}, ex::require(ex::determinism::run_to_run)};
cub::DeviceScan::InclusiveSum(d_in, d_out, n, env);
```

`ex::determinism::gpu_to_gpu` gives the same bits across GPU architectures (slower). It works for
float `DeviceReduce::Sum`; for `DeviceScan` only with integral types (a static_assert otherwise),
so scan fixed-point `int64_t` values there. `double` alone does not make bits match.

## Newer algorithms worth knowing (CCCL 3.x)

- **`cub::DeviceTopK::MaxPairs` / `MinPairs` / `MaxKeys` / `MinKeys`**: 16x faster than a full sort.
  It only compiles with this environment (a plain stream is a static_assert) and the output is unsorted.

  ```cpp
  auto env = cuda::std::execution::env{
    cuda::stream_ref{stream}, ex::require(ex::determinism::not_guaranteed, ex::output_ordering::unsorted)};

  // Values are the indices 0, 1, 2, ..., so d_top_ids receives the positions of the top k scores.
  cub::DeviceTopK::MaxPairs(d_scores, d_top_scores, cuda::make_counting_iterator(0), d_top_ids, n, k, env);
  ```

- `cub::DeviceFor::ForEachInExtents(cuda::std::extents<int, 512, 512, 256>{}, op)`: a loop over a
  grid with no kernel. `op` gets the linear index first: `(int idx, int i, int j, int k)`.
- `cub::DeviceSegmentedScan` (scan per CSR segment), `thrust::unique_count`, `thrust::reduce_into`.

## CUB block and warp primitives

The building blocks for when you do need a custom kernel:

```cpp
template <int BLOCK_THREADS> // must equal blockDim.x at launch
__global__ void center(const float* in, float* out)
{
  using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;
  __shared__ typename BlockReduce::TempStorage temp;
  __shared__ float mean;

  const int i     = blockIdx.x * BLOCK_THREADS + threadIdx.x;
  const float sum = BlockReduce(temp).Sum(in[i]); // valid in thread 0 only
  if (threadIdx.x == 0)
  {
    mean = sum / BLOCK_THREADS;
  }
  __syncthreads(); // publishes mean, and frees temp for reuse

  out[i] = in[i] - mean;
}

// Launch with the same constant.
center<256><<<num_blocks, 256>>>(d_in, d_out);
```

Three things go wrong silently: `BLOCK_THREADS` not matching the launch, reusing `TempStorage`
without a `__syncthreads()` in between, and reading a reduction result outside thread or lane 0.

## libcu++

- In device code use `cuda::std::` (`array`, `span`, `optional`, `tuple`, `numeric_limits`) instead of
  `std::`, and the helpers in `<cuda/cmath>` (`ceil_div`, `round_up`). Mark lambdas `__host__ __device__`.
- **Standard parallel algorithms on the GPU** (CCCL 3.4+): `cuda::std::` algorithms with the
  `cuda::execution::gpu` policy, the only one with a backend. Iterators must be random access
  into device memory. Headers: `<cuda/std/execution>`, `<cuda/std/algorithm>`, `<cuda/std/numeric>`.
  ```cpp
  #include <cuda/std/algorithm>
  #include <cuda/std/execution>
  #include <cuda/std/numeric>
  #include <cuda/stream>

  // stream can be a cuda::stream, a cuda::stream_ref or a cudaStream_t.
  auto policy = cuda::execution::gpu.with(cuda::get_stream, stream);

  float sum = cuda::std::reduce(policy, v.begin(), v.end(), 0.0f);

  cuda::std::transform(policy, v.begin(), v.end(), out.begin(), [] __host__ __device__(float x) {
    return 2.0f * x;
  });
  ```
- **Runtime API**: `cuda::stream`, `cuda::make_buffer`, `cuda::launch`, `cuda::copy_bytes` replace
  `cudaStreamCreate`, `cudaMalloc`, and `<<<>>>`. Full example in `references/libcudacxx.md`.

## References

- Docs (development branch): https://nvidia.github.io/cccl/unstable/ with pages
  `thrust/algorithms.html`, `cub/device_wide.html`, `cub/block_wide.html`, `cub/environment.html`,
  `cub/determinism.html`, `libcudacxx/runtime.html`, and `cccl/3.0_migration_guide.html`
- If you can clone repositories, clone https://github.com/NVIDIA/cccl into a temporary directory
  to read headers, the documented examples in `cub/test/*_api.cu`, and `examples/`.
