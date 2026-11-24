// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Runs 8 independent signal channels, each on its own CUDA stream so they can overlap.
// Every step smooths the channel in place and records its energy (sum of squares);
// the energies are only reported at the end.
// Build: nvcc -O2 -std=c++17 -arch=native --extended-lambda pipeline.cu -o pipeline

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>

#include <chrono>
#include <cstdio>
#include <vector>

int main()
{
  const int n_channels = 8;
  const int n          = 1 << 15; // samples per channel
  const int steps      = 400;

  std::vector<cudaStream_t> streams(n_channels);
  std::vector<thrust::device_vector<float>> sig(n_channels);
  for (int c = 0; c < n_channels; ++c)
  {
    cudaStreamCreateWithFlags(&streams[c], cudaStreamNonBlocking);
    sig[c].resize(n);
    thrust::transform(
      thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(n), sig[c].begin(), [c] __device__(int i) {
        return ((i * 37 + c * 11) % 101) * 0.01f;
      });
  }
  std::vector<std::vector<float>> energy(n_channels, std::vector<float>(steps));
  cudaDeviceSynchronize();

  auto t0 = std::chrono::steady_clock::now();
  for (int s = 0; s < steps; ++s)
  {
    for (int c = 0; c < n_channels; ++c)
    {
      auto policy = thrust::cuda::par.on(streams[c]);
      thrust::transform(policy, sig[c].begin(), sig[c].end(), sig[c].begin(), [] __device__(float x) {
        return 0.98f * x + 0.01f;
      });
      energy[c][s] = thrust::transform_reduce(
        policy,
        sig[c].begin(),
        sig[c].end(),
        [] __host__ __device__(float x) {
          return x * x;
        },
        0.0f,
        thrust::plus<float>());
    }
  }
  cudaDeviceSynchronize();
  auto t1 = std::chrono::steady_clock::now();

  double checksum = 0;
  for (int c = 0; c < n_channels; ++c)
  {
    for (int s = 0; s < steps; ++s)
    {
      checksum += energy[c][s];
    }
  }
  std::printf("checksum=%.3f last_energy_ch0=%.4f\n", checksum, energy[0][steps - 1]);
  std::printf("elapsed: %.1f ms\n", std::chrono::duration<double, std::milli>(t1 - t0).count());
  return 0;
}
