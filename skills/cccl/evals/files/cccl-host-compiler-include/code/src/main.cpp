// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "gpu_stats.h"

void print_report(const FleetStats& s);

int main()
{
  thrust::host_vector<float> readings = make_readings(1 << 20);
  FleetStats s                        = gpu_fleet_stats({readings.data(), readings.size()});
  print_report(s);
  return 0;
}
