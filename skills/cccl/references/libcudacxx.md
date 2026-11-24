# libcu++ beyond `cuda::std`

Everything here ships in the CUDA 13.4 toolkit (CCCL 3.5) under `<ctk>/include/cccl/cuda/`.
Older toolkits have less of it: check `CCCL_VERSION` and compile a small test first.

## Runtime API: streams, buffers, kernel launch

RAII replacements for `cudaStreamCreate`, `cudaMalloc`, and `kernel<<<grid, block>>>(...)`.
Errors are reported as `cuda::cuda_error` exceptions instead of return codes.

```cpp
// nvcc -O2 -std=c++17 -arch=native saxpy.cu -o saxpy
#include <cuda/algorithm>     // cuda::copy_bytes
#include <cuda/buffer>        // cuda::make_buffer
#include <cuda/cmath>         // cuda::ceil_div
#include <cuda/devices>       // cuda::devices
#include <cuda/launch>        // cuda::launch, make_config, grid_dims, block_dims
#include <cuda/memory_pool>   // cuda::device_default_memory_pool
#include <cuda/std/span>
#include <cuda/stream>

#include <cstdio>
#include <vector>

struct saxpy
{
  float a;
  // The config is passed first when the functor accepts it; it lets hierarchy queries
  // use the statically known block size.
  template <class Config>
  __device__ void operator()(Config config, cuda::std::span<const float> x, cuda::std::span<float> y) const
  {
    const auto i = cuda::gpu_thread.rank(cuda::grid, config);   // global thread index
    if (i < y.size())
    {
      y[i] = a * x[i] + y[i];
    }
  }
};

int main()
{
  const int n          = 1 << 24;
  cuda::device_ref dev = cuda::devices[0];
  cuda::stream stream{dev};
  auto pool = cuda::device_default_memory_pool(dev);

  auto x = cuda::make_buffer<float>(stream, pool, n, 1.0f);            // filled with 1
  auto y = cuda::make_buffer<float>(stream, pool, n, 2.0f);
  // cuda::make_buffer<float>(stream, pool, n, cuda::no_init) skips the fill

  auto config = cuda::make_config(cuda::grid_dims(cuda::ceil_div(n, 256)), cuda::block_dims<256>());
  cuda::launch(stream, config, saxpy{3.0f}, cuda::std::span<const float>(x), cuda::std::span<float>(y));

  std::vector<float> host(n);
  cuda::copy_bytes(stream, y, host);
  stream.sync();
  std::printf("y[100] = %g\n", host[100]);   // 5
}
```

- `cuda::launch` needs something to submit work to (a `cuda::stream` or `cuda::stream_ref`).
  It does not wait; call `stream.sync()` before reading results on the host.
- A buffer converts to `cuda::std::span<T>`; `buffer.data()` gives the raw device pointer for
  CUB and Thrust calls, and `cuda::stream_ref{stream}` or `stream.get()` gives them the stream.
- `cuda::gpu_thread.index(cuda::block)` and `.rank(...)` / `.count(...)` replace
  `threadIdx` / `blockIdx` arithmetic (`<cuda/hierarchy>`).
- Lambdas work as kernels too: `[] __device__ (cuda::std::span<float> y) { ... }` with
  `--extended-lambda`.

## Standard parallel algorithms on the GPU

`cuda::std::` algorithms with the `cuda::execution::gpu` policy (CCCL 3.4+). Useful for porting
`std::execution::par` code without switching to Thrust names.

```cpp
#include <cuda/std/algorithm>
#include <cuda/std/execution>
#include <cuda/std/numeric>
#include <cuda/memory_pool>
#include <cuda/stream>

cuda::stream stream{cuda::devices[0]};
auto pool   = cuda::device_default_memory_pool(cuda::devices[0]);
auto policy = cuda::execution::gpu.with(cuda::get_stream, stream)
                                  .with(cuda::mr::get_memory_resource, pool);

double sumsq = cuda::std::transform_reduce(policy, v.begin(), v.end(), 0.0, cuda::std::plus<>{},
                 [] __host__ __device__(float x) { return double(x) * x; });
auto n_big   = cuda::std::count_if(policy, v.begin(), v.end(),
                 [] __host__ __device__(float x) { return x > 1.0f; });
cuda::std::inclusive_scan(policy, v.begin(), v.end(), out.begin());
```

- The plain `cuda::execution::gpu` policy uses the default stream.
- `.with(cuda::get_stream, s)` accepts a `cuda::stream`, a `cuda::stream_ref`, or a `cudaStream_t`.
- Only `cuda::execution::gpu` has a backend. `cuda::std::execution::par` and `seq` with these
  overloads fail with a static_assert ("requires at least one selected backend").
- Iterators must be random access and point to device memory: raw device pointers,
  `thrust::device_vector` iterators, libcu++ fancy iterators. Host containers are not copied
  for you.
- Compile the file with nvcc. Callables must be device-callable.
- The set of supported algorithms grows with each release; compile a small test for the one you
  need, or use the Thrust equivalent.

## Utilities worth using instead of hand-rolled helpers

| Header | What |
|---|---|
| `<cuda/cmath>` | `ceil_div`, `round_up`, `round_down`, `ilog2`, `ilog10`, `ipow`, `isqrt`, `is_power_of_two`, `next_power_of_two`, `uabs`, `mul_hi` |
| `<cuda/numeric>` | `add_overflow`, `sub_overflow`, `mul_overflow`, `div_overflow` (return `overflow_result<T>{value, overflow}`), `overflow_cast<T>`, `narrow<T>`, saturating variants |
| `<cuda/bit>` | `bitmask`, `bitfield_insert`, `bitfield_extract`, `bit_reverse` |
| `<cuda/functional>` | `minimum`, `maximum`, `proclaim_return_type`, `proclaim_copyable_arguments` |
| `<cuda/iterator>` | `counting_iterator`, `constant_iterator`, `transform_iterator`, `zip_iterator`, `zip_transform_iterator`, `strided_iterator`, `permutation_iterator`, `discard_iterator`, `tabulate_output_iterator`, `transform_output_iterator`, `shuffle_iterator` |
| `<cuda/std/...>` | Device-usable `array`, `span`, `mdspan`, `optional`, `variant`, `expected`, `tuple`, `complex`, `bit`, `numeric_limits`, `algorithm`, `atomic` |
| `<cuda/warp>` | `warp_shuffle`, `warp_match_any`, `warp_match_all` |

All of these are `constexpr` where possible and callable from both host and device code.
