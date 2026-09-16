---
name: "cudaq-importing"
title: "CUDA-Q Importing"
description: "Use when porting circuits from another framework (e.g. Qiskit) into CUDA-Q kernels while preserving the source algorithm and validation fidelity."
version: "1.0.2"
author: "CUDA-Q Team <cuda-quantum@nvidia.com>"
tags: [cuda-quantum, quantum-computing, importing, porting, migration, qiskit, kernels, nvidia]
tools: [Read, Glob, Grep]
license: "Apache-2.0"
compatibility: "Python 3.10+"
metadata:
    author: "CUDA-Q Team <cuda-quantum@nvidia.com>"
    short-description: "Port circuits from other frameworks into CUDA-Q"
    tags:
        - cuda-quantum
        - quantum-computing
        - importing
        - porting
        - migration
        - qiskit
        - nvidia
    languages:
        - python
    domain: "quantum"
---

# CUDA-Q Importing

## Purpose

Use this skill to port quantum circuits from another framework into CUDA-Q
Python kernels. This includes Qiskit code and Qiskit-style circuit construction,
as well as other framework-driven circuit builders. The goal is a framework-free
CUDA-Q port that preserves the source quantum algorithm, matches source behavior
at small test sizes, and documents any unavoidable CUDA-Q limitations.

For authoring new CUDA-Q kernels from scratch, and for CUDA-Q installation,
simulation targets, QPU access, and parallelization, use the `cudaq-guide`
skill (`/cudaq-guide author` for kernel authoring).

## Prerequisites

- Python 3.10+.
- CUDA-Q installed in the target environment. Check the runtime with:
  `python -c "import cudaq; print(getattr(cudaq, '__version__', 'unknown'))"`.
- Access to the source implementation and a way to run or inspect its expected
  behavior.
- To validate against the source framework (e.g. Qiskit/Aer), it must be
  installed in the validation environment only. The final CUDA-Q port itself
  must not require the source framework.
- When using CUDA-Q documentation or repository MCP connectors, verify the
  connector is available before relying on it; otherwise use local docs or the
  source tree.
- When debugging and the installed CUDA-Q version differs from the latest
  documentation, review relevant documentation or source changes before
  treating a behavior difference as a porting bug.

## Workflow

1. Read the source circuit construction and identify the exact algorithm,
   qubit/register layout, measurement behavior, and any framework helpers.
2. Preserve the high-level quantum algorithm. Do not replace mid-circuit
   measurement, QPE structure, oracle definitions, or decomposition strategy
   without explicit user permission.
3. Select the CUDA-Q execution pattern:
   - Use `cudaq.sample` for final-measurement sampling.
   - Use `cudaq.run` when mid-circuit measurement values must be returned or
     used per shot.
   - Use runtime-argument kernels instead of generated per-size kernels unless
     CUDA-Q requires a fixed-length return shape.
4. Translate gates and subcircuits. For detailed gate mappings, ordering rules,
   precision guidance, and helper-extraction patterns, read
   [references/porting-reference.md](references/porting-reference.md).
5. Remove runtime source-framework dependencies from the CUDA-Q port. Extract
   pure helpers into framework-free modules.
6. Validate with small deterministic inputs before scaling. Compare raw count
   keys and distributions, not just aggregate fidelity.
7. Re-run any previously failing configurations after every fix.

## Core Rules

- Keep the source algorithm intact unless the user approves a change.
- Do not introduce fixed qubit caps, fixed control arities, or source-framework
  imports unless they are genuinely unavoidable and documented.
- Prefer native CUDA-Q gates (`r1.ctrl`, `x.ctrl`, `swap.ctrl`, etc.) over
  transpiling through the source framework.
- Keep bit-order conversion at the port boundary: allocation order,
  measurement return list, or final count-key formatting.
- Match floating-point precision when comparing CUDA-Q and source results if
  fidelity differences matter (CUDA-Q defaults to fp32, Qiskit to fp64).
- Accept source flags that become no-ops in CUDA-Q when doing so preserves
  source-compatible behavior.

## When to Read the Reference

Read [references/porting-reference.md](references/porting-reference.md) when
you need any of the following:

- Qiskit-to-CUDA-Q gate translation table.
- Bit-ordering and count-key conventions.
- CUDA-Q fp32 vs Qiskit fp64 precision implications.
- Pure-Python helper extraction and import-blocker validation.
- Recursive-constructor emitters or gate-recorder patterns.
- Detailed port validation checklist and external CUDA-Q references.

## Limitations

- Guidance targets CUDA-Q 0.14/0.15 decorator-mode Python APIs. Re-check
  behavior against the installed CUDA-Q version for version-sensitive features.
- Some CUDA-Q kernel-language constructs are constrained compared with normal
  Python; use the companion `cudaq-guide` skill (`/cudaq-guide author`) for core
  CUDA-Q authoring constraints and shared kernel patterns.
- CUDA-Q and source frameworks differ in default precision and count-key display
  order. Apparent fidelity or bitstring mismatches may be convention
  differences.
- Hardware-target behavior, available backends, and target options depend on
  the local CUDA-Q installation.
- This skill does not guarantee equivalent performance; it focuses on
  correctness-preserving ports.

## Troubleshooting

Use this format when diagnosing failures:

- **Error:** `ModuleNotFoundError: qiskit` (or another source framework) from a
  CUDA-Q path.
  **Cause:** The port still imports the source framework.
  **Solution:** Move pure helpers into a framework-free module and verify with
  the import-blocker pattern in the reference.

- **Error:** Fidelity looks plausible but raw keys are reversed.
  **Cause:** The source framework and CUDA-Q count-key ordering differ.
  **Solution:** Fix allocation, return-list order, or formatting at the port
  boundary. Do not alter the algorithm.

- **Error:** Deep-circuit fidelity differs between frameworks.
  **Cause:** CUDA-Q and the source framework may be using different
  floating-point precision.
  **Solution:** Match precision before comparing, then rerun the smallest
  failing deterministic case.

- **Error:** A multi-controlled operation works for small controls but fails or
  silently changes behavior at higher arity.
  **Cause:** The port used a fixed-arity dispatcher.
  **Solution:** Use CUDA-Q control-list patterns for arbitrary arity.

- **Error:** MCP documentation or repository lookup fails.
  **Cause:** Connector unavailable, stale, or transiently failing.
  **Solution:** Verify the connector/resource list, retry transient failures
  once, then fall back to local docs/source or official CUDA-Q docs. Do not
  change the port based on unverified MCP results.

- **Error:** CUDA-Q behavior conflicts with documentation while debugging.
  **Cause:** The installed CUDA-Q version may differ from the latest
  documentation.
  **Solution:** Check `cudaq.__version__`, then review relevant documentation or
  source changes between the installed version and latest before changing the
  port.

## References

- [Detailed porting reference](references/porting-reference.md)
- Companion skill: `cudaq-guide` (`/cudaq-guide author`) for CUDA-Q authoring
  patterns, kernel-language constraints, execution APIs, and debugging workflow.
