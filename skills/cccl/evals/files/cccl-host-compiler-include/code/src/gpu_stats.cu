// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>

#include "gpu_stats.h"

FleetStats gpu_fleet_stats(cuda::std::span<const float> readings)
{
  thrust::device_vector<float> d(readings.begin(), readings.end());
  double sum = thrust::reduce(d.begin(), d.end(), 0.0);
  float mx   = *thrust::max_element(d.begin(), d.end());
  return {sum / static_cast<double>(d.size()), mx};
}

thrust::host_vector<float> make_readings(int n)
{
  thrust::host_vector<float> h(n);
  for (int i = 0; i < n; ++i)
  {
    h[i] = static_cast<float>((i * 7919u) % 1000u) * 0.1f;
  }
  return h;
}
