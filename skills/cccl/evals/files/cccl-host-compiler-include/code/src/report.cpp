// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cstdio>

#include "gpu_stats.h"

void print_report(const FleetStats& s)
{
  std::printf("fleet mean=%.3f max=%.1f\n", s.mean, s.max);
}
