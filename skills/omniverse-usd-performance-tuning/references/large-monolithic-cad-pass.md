# Large Monolithic Repeated-CAD Optimization Pass

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Execution contract for large monolithic CAD-style stages with many repeated
meshes and low or no instance/prototype use, when the user asks for the safest
useful optimization before decimation. Referenced from the
`omniverse-usd-performance-tuning` SKILL.md *Optimization ordering* section.

- Treat lossless hierarchy/geometry deduplication or prototype/reference
  restructuring as the primary optimization. A USDC/crate repack is only a
  secondary packaging step and is not sufficient by itself — and a repack is the
  free re-encode any export achieves, so its disk delta must be attributed to
  re-encoding, not to the optimization (see the repack-normalized footprint
  facts below).
- Do not run decimation, primitive fitting, quantization, fuzzy matching, or
  topology edits unless the user explicitly approves that lossy operation.
- Write a separate optimized stage; never overwrite the source unless the user
  explicitly approves in-place mutation.

- For large or binary/crate USD artifacts, use binary-safe file operations (for example shell copy/export tools or Python opened in `rb`/`wb` mode). Do not pass byte content to text-only write APIs.
- Record baseline and after metrics for file size, prim count, mesh count,
  repeated mesh families, affected mesh prims, authored references, payloads,
  instanceable/prototype usage, and validation status.
- Run targeted before/after validation such as open/load checks, the
  minimum-openability pass owned by `usd-validation-runner`, and affected-prim
  composition checks. Treat expensive whole-stage equivalence, visibility,
  duplicate-family, or exhaustive composition sweeps as full-sweep validation;
  for large CAD stages, skip or defer those unless explicitly requested. Do not
  describe a minimum-openability log as "full validation"; it is checker
  evidence, not the full-sweep policy.
- In the final response, include a compact "large-stage policy" ledger with
  these exact facts: baseline is a large monolithic CAD-style repeated-mesh
  stage; baseline authored references/payloads/instanceable or prototype
  counts; operation order; optimized output path; source-not-overwritten
  status; mesh/prim count before and after; repeated-family and affected-mesh
  counts; instanceable/prototype/reference changes; targeted before/after
  validation evidence; and
  `full_sweep_validation: skipped/deferred due to large-stage policy`.
- Report footprint honestly with all three repack-normalized sizes: raw input,
  the repack-normalized baseline (input losslessly re-crated to the same
  encoding, zero dedupe), and optimized — and attribute the split (`X% is
  re-encoding; Y% is the structural optimization`). Score the structural win
  against the normalized baseline, not the raw input. Populate the report's
  `footprint` block; presenting the repack delta as the optimization win fails
  the report gate.
- Also state which runtime metrics were not measured. Do not claim FPS, VRAM,
  Hydra, RTX, renderer, or draw-call wins unless those metrics were actually
  captured.
