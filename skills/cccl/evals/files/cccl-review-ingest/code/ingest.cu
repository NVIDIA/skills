// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Sensor ingest: validates readings, compacts the valid ones per sensor, sorts the
// per-reading drift by magnitude bucket, and computes each sensor's minimum latency.
// Build: nvcc -O2 -std=c++17 -arch=native --extended-lambda ingest.cu -o ingest

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_reduce.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>

int main()
{
  const int n_sensors  = 50000;
  const int per_sensor = 64;
  const int n_readings = n_sensors * per_sensor;
  const float inf      = std::numeric_limits<float>::infinity();

  // Raw readings: latency in ms (+inf when the packet was lost) and signed drift in ppm
  thrust::device_vector<float> latency(n_readings);
  thrust::device_vector<int> drift(n_readings);
  thrust::transform(
    thrust::counting_iterator<unsigned>(0),
    thrust::counting_iterator<unsigned>(n_readings),
    latency.begin(),
    [inf] __device__(unsigned i) {
      unsigned h = i * 2654435761u;
      // sensors whose id is divisible by 97 are offline: every packet lost
      bool lost = ((i / 64u) % 97u == 0u) || (h % 11u == 0u);
      return lost ? inf : 1.0f + static_cast<float>(h >> 22);
    });
  thrust::transform(
    thrust::counting_iterator<unsigned>(0),
    thrust::counting_iterator<unsigned>(n_readings),
    drift.begin(),
    [] __device__(unsigned i) {
      unsigned h = (i ^ 0x5bd1e995u) * 2246822519u;
      return static_cast<int>(h % 2001u) - 1000; // [-1000, 1000]
    });

  // 1. Valid readings
  thrust::device_vector<bool> valid(n_readings);
  thrust::transform(latency.begin(), latency.end(), valid.begin(), [] __device__(float l) {
    return l < 1.0e30f;
  });
  int n_valid = thrust::reduce(valid.begin(), valid.end());

  // 2. Valid readings per sensor (each sensor has 64 slots) and each sensor's output offset
  thrust::device_vector<std::uint8_t> valid_per_sensor(n_sensors);
  const bool* v = thrust::raw_pointer_cast(valid.data());
  thrust::transform(
    thrust::counting_iterator<int>(0),
    thrust::counting_iterator<int>(n_sensors),
    valid_per_sensor.begin(),
    [v] __device__(int s) {
      std::uint8_t c = 0;
      for (int j = 0; j < 64; ++j)
      {
        c += v[s * 64 + j];
      }
      return c;
    });
  thrust::device_vector<int> sensor_offset(n_sensors);
  thrust::exclusive_scan(valid_per_sensor.begin(), valid_per_sensor.end(), sensor_offset.begin());

  // 3. Sort drift values; only the low bits matter since |drift| <= 1000
  thrust::device_vector<int> drift_alt(n_readings);
  cub::DoubleBuffer<int> keys(thrust::raw_pointer_cast(drift.data()), thrust::raw_pointer_cast(drift_alt.data()));
  const int key_bits = 11; // 2^11 = 2048 > 2001 distinct values
  size_t temp_bytes  = 0;
  cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, keys, n_readings, 0, key_bits);
  thrust::device_vector<char> temp(temp_bytes);
  cub::DeviceRadixSort::SortKeys(thrust::raw_pointer_cast(temp.data()), temp_bytes, keys, n_readings, 0, key_bits);
  int median_drift = drift[n_readings / 2];

  // 4. Minimum latency per sensor; offline sensors must stay +inf
  thrust::device_vector<int> seg_offsets(n_sensors + 1);
  thrust::transform(
    thrust::counting_iterator<int>(0),
    thrust::counting_iterator<int>(n_sensors + 1),
    seg_offsets.begin(),
    [] __device__(int s) {
      return s * 64;
    });
  thrust::device_vector<float> min_latency(n_sensors);
  const int* so = thrust::raw_pointer_cast(seg_offsets.data());
  size_t temp2  = 0;
  cub::DeviceSegmentedReduce::Min(nullptr, temp2, latency.begin(), min_latency.begin(), n_sensors, so, so + 1);
  thrust::device_vector<char> temp_min(temp2);
  cub::DeviceSegmentedReduce::Min(
    thrust::raw_pointer_cast(temp_min.data()), temp2, latency.begin(), min_latency.begin(), n_sensors, so, so + 1);
  int offline = thrust::count_if(min_latency.begin(), min_latency.end(), [] __device__(float m) {
    return isinf(m);
  });

  int last_offset = sensor_offset[n_sensors - 1];
  std::printf("valid readings:          %d of %d\n", n_valid, n_readings);
  std::printf("last sensor offset:      %d\n", last_offset);
  std::printf("median drift (ppm):      %d\n", median_drift);
  std::printf("offline sensors:         %d\n", offline);
  return 0;
}
