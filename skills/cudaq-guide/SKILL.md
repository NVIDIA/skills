---
name: "cudaq-guide"
title: "CUDA-Q Guide"
description: "Use for CUDA-Q setup, simulation targets, QPU access, and @cudaq.kernel authoring guidance."
version: "1.1.2"
author: "CUDA-Q Team <cuda-quantum@nvidia.com>"
tags: [cuda-quantum, quantum-computing, onboarding, getting-started, authoring, kernels, nvidia]
tools: [Read, Glob, Grep]
license: "Apache-2.0"
compatibility: "Python 3.10+, C++ 20"
metadata:
    author: "CUDA-Q Team <cuda-quantum@nvidia.com>"
    tags:
        - cuda-quantum
        - quantum-computing
        - onboarding
        - getting-started
        - nvidia
    languages:
        - python
        - c++
    domain: "quantum"
---

# CUDA-Q Guide

## Purpose

Guide users through CUDA-Q installation, basic kernels, GPU simulation targets,
QPU access, built-in applications, multi-GPU execution, and Python
`@cudaq.kernel` authoring. For Qiskit-to-CUDA-Q ports, route to the
`cudaq-importing` skill instead.

## Prerequisites

- Python 3.10+ for Python CUDA-Q workflows.
- CUDA Toolkit and an NVIDIA GPU for GPU-accelerated targets on Linux.
- CPU-only simulation is available through `qpp-cpu`; macOS is CPU-only.
- C++ workflows require Linux or WSL and C++20.
- QPU workflows require provider-specific credentials and accounts.

## Instructions

- Invoke with `/cudaq-guide [argument]`.
- If no argument is given, display the onboarding menu and ask which topic the
  user wants.
- Use the routing table below to choose the relevant reference file.
- Read local CUDA-Q documentation files when the answer depends on a specific
  CUDA-Q version or backend behavior.
- Do not answer Qiskit porting questions from this skill; use
  `cudaq-importing`.

## Routing by Argument

| Argument | Action | Reference |
|---|---|---|
| `install` | Walk through Python or C++ installation and validation. | [references/onboarding.md](references/onboarding.md) |
| `test-program` | Build and run a Bell-state kernel. | [references/onboarding.md](references/onboarding.md) |
| `gpu-sim` | Select GPU, multi-GPU, tensor-network, or CPU targets. | [references/onboarding.md](references/onboarding.md) |
| `qpu` | Guide provider selection and credential-safe QPU setup. | [references/onboarding.md](references/onboarding.md) |
| `applications` | Summarize CUDA-Q application areas and notebooks. | [references/onboarding.md](references/onboarding.md) |
| `parallelize` | Choose `mgpu`, `mqpu`, async dispatch, or distributed observe. | [references/onboarding.md](references/onboarding.md) |
| `author` | Author CUDA-Q Python kernels, select execution APIs, and debug compiler issues. | [references/authoring.md](references/authoring.md) |
| _(none)_ | Print the menu below and ask which topic to explore. | This file |

## Menu

```text
CUDA-Q Getting Started

CUDA-Q is NVIDIA's unified quantum-classical programming model for CPUs, GPUs, and QPUs.
Supports Python and C++. Docs: https://nvidia.github.io/cuda-quantum/latest/

Choose a topic:
  /cudaq-guide install         Install CUDA-Q
  /cudaq-guide test-program    Write and run a Bell-state kernel
  /cudaq-guide gpu-sim         Accelerate simulation on NVIDIA GPUs
  /cudaq-guide qpu             Connect to real QPU hardware
  /cudaq-guide applications    Explore what you can build
  /cudaq-guide parallelize     Run across GPUs or QPUs
  /cudaq-guide author          Author @cudaq.kernel Python code
```

## Reference Files

- [references/onboarding.md](references/onboarding.md): installation, test
  program, GPU targets, QPU providers, application areas, parallelization
  modes, examples, and platform troubleshooting.
- [references/authoring.md](references/authoring.md): execution APIs,
  kernel-language constraints, silent-failure pitfalls, recurring coding
  patterns, resource metrics, debugging, and validation.

## Limitations

- Guidance targets CUDA-Q Python/C++ workflows, with authoring details focused
  on decorator-mode Python APIs used in CUDA-Q 0.14 and 0.15.
- GPU and multi-GPU support depends on local CUDA-Q, CUDA Toolkit, driver, MPI,
  and hardware availability.
- QPU access and target options are provider-specific and may change; verify
  against local docs before giving operational steps.

## Troubleshooting

- **Import error after `pip install cudaq`:** check Python 3.10+ and supported
  OS.
- **No GPU detected:** verify CUDA Toolkit and `nvidia-smi`; fall back to
  `qpp-cpu`.
- **Kernel compile error:** read [references/authoring.md](references/authoring.md)
  and check the restricted kernel-language subset.
- **Version-specific behavior differs:** compare `cudaq.__version__` with the
  latest documentation, then review relevant documentation or source changes
  when debugging an installed version that is not the latest release.
- **QPU submission fails:** verify provider credentials are set as environment
  variables or through a secrets manager, never hardcoded.
- **Documentation lookup fails:** retry transient MCP or repository lookup once,
  then fall back to local docs or official CUDA-Q documentation.
