// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Summary statistics over sensor readings. Built fine with CUDA 12.2.
// Build: nvcc -O2 -std=c++17 -arch=native stats.cu -o stats

#include <cub/cub.cuh>

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <cstdio>
#include <vector>

#define CHECK(x)                                                                          \
  do                                                                                      \
  {                                                                                       \
    cudaError_t e = (x);                                                                  \
    if (e != cudaSuccess)                                                                 \
    {                                                                                     \
      std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
      return 1;                                                                           \
    }                                                                                     \
  } while (0)

struct calibrate : public thrust::unary_function<float, float>
{
  float gain, offset;
  __host__ __device__ float operator()(float raw) const
  {
    return raw * gain + offset;
  }
};

struct reading_at : public thrust::unary_function<int, float>
{
  __host__ __device__ float operator()(int i) const
  {
    return static_cast<float>((i * 37) % 1000) * 0.01f;
  }
};

int main()
{
  const int n = 1 << 20;
  thrust::device_vector<float> raw(n);
  thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n), raw.begin(), reading_at());

  thrust::device_vector<float> cal(n);
  thrust::transform(raw.begin(), raw.end(), cal.begin(), calibrate{2.0f, -1.0f});

  // Pass-through copy kept for the archive buffer
  thrust::device_vector<float> archive(n);
  thrust::transform(cal.begin(), cal.end(), archive.begin(), thrust::identity<float>());

  const float* d_in = thrust::raw_pointer_cast(cal.data());
  thrust::device_vector<float> out(3);
  float* d_out = thrust::raw_pointer_cast(out.data());

  // Temp storage sized for the largest of the three reductions
  size_t bytes_sum = 0, bytes_max = 0, bytes_min = 0;
  CHECK(cub::DeviceReduce::Reduce(nullptr, bytes_sum, d_in, d_out + 0, n, cub::Sum(), 0.0f, 0, false));
  CHECK(cub::DeviceReduce::Reduce(nullptr, bytes_max, d_in, d_out + 1, n, cub::Max(), -1e30f, 0, false));
  CHECK(cub::DeviceReduce::Reduce(nullptr, bytes_min, d_in, d_out + 2, n, cub::Min(), 1e30f, 0, false));
  size_t temp_bytes = CUB_MAX(bytes_sum, CUB_MAX(bytes_max, bytes_min));
  void* d_temp      = nullptr;
  CHECK(cudaMalloc(&d_temp, temp_bytes));

  CHECK(cub::DeviceReduce::Reduce(d_temp, temp_bytes, d_in, d_out + 0, n, cub::Sum(), 0.0f, 0, false));
  CHECK(cub::DeviceReduce::Reduce(d_temp, temp_bytes, d_in, d_out + 1, n, cub::Max(), -1e30f, 0, false));
  CHECK(cub::DeviceReduce::Reduce(d_temp, temp_bytes, d_in, d_out + 2, n, cub::Min(), 1e30f, 0, false));

  // Number of 256-element pages needed to store the archive
  int pages = CUB_QUOTIENT_CEILING(n, 256);

  thrust::host_vector<float> h = out;
  std::printf("mean=%.4f max=%.4f min=%.4f pages=%d archived=%d\n", h[0] / n, h[1], h[2], pages, (int) archive.size());
  CHECK(cudaFree(d_temp));
  return 0;
}
