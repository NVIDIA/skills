// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Bins particles into grid cells every simulation step by sorting them by cell id.
// The grid has 256 x 256 cells, so cell ids are in [0, 65536).
// Build: nvcc -O2 -std=c++17 -arch=native --extended-lambda bin_particles.cu -o bin_particles

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include <cstdio>

struct Particle
{
  unsigned cell;
  float x, y, z;
  float vx, vy, vz;
};

__host__ __device__ inline unsigned cell_of(float x, float y)
{
  unsigned cx = static_cast<unsigned>(x * 256.0f) & 255u;
  unsigned cy = static_cast<unsigned>(y * 256.0f) & 255u;
  return cy * 256u + cx;
}

int main()
{
  const int n     = 1 << 23; // 8M particles
  const int steps = 10;

  thrust::device_vector<Particle> particles(n);
  thrust::transform(
    thrust::counting_iterator<unsigned>(0),
    thrust::counting_iterator<unsigned>(n),
    particles.begin(),
    [] __device__(unsigned i) {
      unsigned h1 = i * 2654435761u, h2 = (i ^ 0x9e3779b9u) * 2246822519u;
      Particle p;
      p.x    = (h1 >> 8) / 16777216.0f;
      p.y    = (h2 >> 8) / 16777216.0f;
      p.z    = 0.5f;
      p.vx   = 0.001f;
      p.vy   = -0.0007f;
      p.vz   = 0.0f;
      p.cell = cell_of(p.x, p.y);
      return p;
    });

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float sort_ms = 0.0f;

  for (int s = 0; s < steps; ++s)
  {
    // advance and re-bin
    thrust::transform(particles.begin(), particles.end(), particles.begin(), [] __device__(Particle p) {
      p.x    = p.x + p.vx - floorf(p.x + p.vx);
      p.y    = p.y + p.vy - floorf(p.y + p.vy);
      p.cell = cell_of(p.x, p.y);
      return p;
    });

    cudaEventRecord(start);
    thrust::sort(particles.begin(), particles.end(), [] __device__(const Particle& a, const Particle& b) {
      return a.cell < b.cell;
    });
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    sort_ms += ms;
  }

  Particle first = particles[0], last = particles[n - 1];
  std::printf("first cell=%u last cell=%u\n", first.cell, last.cell);
  std::printf("avg sort time per step: %.2f ms\n", sort_ms / steps);
  return 0;
}
