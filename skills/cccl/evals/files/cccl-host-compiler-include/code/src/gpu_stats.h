// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <thrust/host_vector.h>

#include <cuda/std/span>

struct FleetStats
{
  double mean;
  float max;
};

FleetStats gpu_fleet_stats(cuda::std::span<const float> readings);
thrust::host_vector<float> make_readings(int n);
