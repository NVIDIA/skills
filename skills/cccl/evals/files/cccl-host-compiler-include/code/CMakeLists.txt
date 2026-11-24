# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.24)
project(fleet_report LANGUAGES CXX CUDA)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_library(fleet_gpu STATIC src/gpu_stats.cu)
target_compile_features(fleet_gpu PUBLIC cxx_std_17 cuda_std_17)

add_executable(fleet_report src/main.cpp src/report.cpp)
target_link_libraries(fleet_report PRIVATE fleet_gpu)
