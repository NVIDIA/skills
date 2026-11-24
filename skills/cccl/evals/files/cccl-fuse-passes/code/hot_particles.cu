// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Per-step diagnostics for a particle simulation: how many particles are "hot" (kinetic
// energy above a threshold), their total energy and the single hottest one.
// Build: nvcc -O2 -std=c++17 -arch=native hot_particles.cu -o hot_particles

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <chrono>
#include <cstdio>

struct Diagnostics
{
  long long hot_count;
  double hot_energy;
  float hottest;
};

struct SumSquares
{
  __host__ __device__ float operator()(float x, float y) const
  {
    return x * x + y * y;
  }
};

struct AddSquare
{
  __host__ __device__ float operator()(float partial, float z) const
  {
    return partial + z * z;
  }
};

struct Kinetic
{
  __host__ __device__ float operator()(float mass, float speed2) const
  {
    return 0.5f * mass * speed2;
  }
};

struct Above
{
  float threshold;
  __host__ __device__ bool operator()(float e) const
  {
    return e > threshold;
  }
};

Diagnostics diagnose(const thrust::device_vector<float>& vx,
                     const thrust::device_vector<float>& vy,
                     const thrust::device_vector<float>& vz,
                     const thrust::device_vector<float>& mass,
                     float threshold)
{
  const size_t n = vx.size();

  thrust::device_vector<float> speed2(n);
  thrust::transform(vx.begin(), vx.end(), vy.begin(), speed2.begin(), SumSquares{});
  thrust::transform(speed2.begin(), speed2.end(), vz.begin(), speed2.begin(), AddSquare{});

  thrust::device_vector<float> energy(n);
  thrust::transform(mass.begin(), mass.end(), speed2.begin(), energy.begin(), Kinetic{});

  thrust::device_vector<float> hot(n);
  auto hot_end = thrust::copy_if(energy.begin(), energy.end(), hot.begin(), Above{threshold});

  Diagnostics d;
  d.hot_count  = hot_end - hot.begin();
  d.hot_energy = thrust::reduce(hot.begin(), hot_end, 0.0);
  d.hottest    = d.hot_count > 0 ? static_cast<float>(*thrust::max_element(hot.begin(), hot_end)) : 0.0f;
  return d;
}

struct Velocity
{
  unsigned salt;
  __host__ __device__ float operator()(unsigned i) const
  {
    unsigned h = (i + salt) * 2654435761u;
    h ^= h >> 15;
    h *= 2246822519u;
    h ^= h >> 13;
    return (static_cast<float>(h % 20001) - 10000.0f) * 1e-3f; // [-10, 10]
  }
};

struct Mass
{
  __host__ __device__ float operator()(unsigned i) const
  {
    return 1.0f + static_cast<float>(i % 7) * 0.5f;
  }
};

int main()
{
  constexpr int n = 32'000'000;
  thrust::device_vector<float> vx(n), vy(n), vz(n), mass(n);
  thrust::counting_iterator<unsigned> idx(0);
  thrust::transform(idx, idx + n, vx.begin(), Velocity{1u});
  thrust::transform(idx, idx + n, vy.begin(), Velocity{77777u});
  thrust::transform(idx, idx + n, vz.begin(), Velocity{4242424u});
  thrust::transform(idx, idx + n, mass.begin(), Mass{});

  const float threshold = 300.0f;
  Diagnostics d         = diagnose(vx, vy, vz, mass, threshold); // warm-up

  constexpr int steps = 20;
  auto t0             = std::chrono::steady_clock::now();
  for (int s = 0; s < steps; ++s)
  {
    d = diagnose(vx, vy, vz, mass, threshold);
  }
  cudaDeviceSynchronize();
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / steps;

  std::printf("hot particles: %lld\n", d.hot_count);
  std::printf("hot energy: %.6e\n", d.hot_energy);
  std::printf("hottest: %.4f\n", d.hottest);
  std::printf("time per step: %.3f ms\n", ms);
  return 0;
}
