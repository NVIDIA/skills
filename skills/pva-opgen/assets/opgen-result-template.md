<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-->

# OpGen result template

This template is a skill workflow.
It is not an NVIDIA SDK output format.

```text
# PVA operator OpGen result

Verdict: PASSED | FAILED | BLOCKED
Reason: <one exact sentence>
Operator: <name>
Project root: <absolute path>
Result time: <UTC ISO 8601>

## Deliverables

- Public API: <paths>
- Host code: <paths>
- Device code: <paths>
- CPU reference: <paths>
- Sample: <paths>
- Build integration: <paths>
- Install target or deployment manifest: <target or path>
- Correctness tests: <paths>
- Performance tests: <paths>
- Documentation: <paths>

## Implementation

- Algorithm: <summary>
- Dataflow: <type and reason>
- Tile and VMEM plan: <summary>
- Vectorization: <vector operations and tail method>
- Arithmetic: <rounding and saturation rules>
- Supported matrix: <cases>
- Limitations: <items or none>

## Verification

- Build-Test verdict: PASSED | FAILED | BLOCKED
- Build-Test result: <absolute path>
- EMU policy: <optional | required>
- EMU result: <PASSED | FAILED | BLOCKED | SKIPPED_OPTIONAL, reason, and log>
- L4T build result: <status and log>
- Hardware correctness: <status and log>
- Hardware measurement: <method and result>

## Debug attempts

- Maximum attempts: <PVA_OPERATOR_MAX_DEBUG_ATTEMPTS>
- Debug calls used: <count, including 0 when none were invoked>
- Stop reason: <verification passed, budget zero or exhausted, blocked, or not reproduced>
- Attempt <number>: <Debug result, report path, source identity, and fresh Build-Test result when repaired>

## Evidence

- SDK-reference lock: <absolute path and SHA-256>
- Local SDK references: <SDK documentation root, Markdown paths, packaged source paths, raw source SHA-256 values, and sections or PDF pages>
- <artifact>: <absolute path> <SHA-256>
```
