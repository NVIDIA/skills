// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Replays an event log. Every event updates a 32-bit register as x -> a * x + b (mod 2^32).
// The net effect of the whole log is again one such update; we compute it on the GPU and
// apply it to the initial register value.
// Build: nvcc -O2 -std=c++17 -arch=native replay.cu -o replay

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <cstdint>
#include <cstdio>
#include <vector>

struct Update
{
  std::uint32_t a;
  std::uint32_t b;
};

// "first, then second": second(first(x))
struct Then
{
  __host__ __device__ Update operator()(const Update& first, const Update& second) const
  {
    return {second.a * first.a, second.a * first.b + second.b};
  }
};

Update net_effect(const thrust::device_vector<Update>& log)
{
  return thrust::reduce(log.begin(), log.end(), Update{1u, 0u}, Then{});
}

int main()
{
  constexpr int num_events = 16'000'000;
  std::vector<Update> h_log(num_events);
  std::uint32_t s = 2463534242u;
  for (auto& e : h_log)
  {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    e = {(s >> 3) | 1u, s * 2654435761u};
  }

  // CPU reference: apply the events one after another.
  std::uint32_t x_ref = 42u;
  for (const auto& e : h_log)
  {
    x_ref = e.a * x_ref + e.b;
  }

  thrust::device_vector<Update> log = h_log;
  Update net                        = net_effect(log);
  std::uint32_t x_gpu               = net.a * 42u + net.b;

  std::printf("register after replay: gpu=%u cpu=%u -> %s\n", x_gpu, x_ref, x_gpu == x_ref ? "MATCH" : "MISMATCH");
  return 0;
}
