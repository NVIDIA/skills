<!-- SPDX-License-Identifier: Apache-2.0 AND CC-BY-4.0 -->
<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

# Validation record — 2026-11-06

Base NVIDIA/skills commit: `c663e5799366d945a70a40b1ab18cedfc817495c`.
API source: cuGraph v26.10.00, `8ff922fabf4521cf540a7ac3cfae857f7edc6f10`.
Internal reference: RAPIDS/rapids-agent-skills `4672643da793932d4c8351f0bea35c724f4c1710`.

Passed locally on macOS (CPU only):

- Four unittest cases: disjoint fixture coverage at 1/2/4/16 ranks including empty ranks; order-independent correct result acceptance; missing/duplicate/nonfinite/wrong-result rejection; existing input directory protection.
- All three Python scripts compile; both GPU runners expose help without importing GPU packages.
- Python code blocks parse; all local Markdown links resolve.
- Eleven evaluation records parse and have unique IDs (ten positive, one negative).
- Skill frontmatter validation and repository skill-card unresolved-marker check.
- Whitespace/diff checks.

Not run: Linux CUDA execution, SG/MG numerical equivalence, Dask runtime, NCCL transport, multi-node launch, failure injection, performance measurement, agent effectiveness benchmark, or NVSkills signing. The cycle fixture checker was tested with synthetic expected outputs; this is not a claim that GPU PageRank produced those outputs.

Initial validation encountered missing PyYAML and a system Python cache path outside the writable workspace. PyYAML was installed in an isolated workspace venv; compilation was rerun with a temporary cache prefix. No system Python configuration was changed.

Catalog submission is not yet eligible: public NVIDIA-owned source placement, release clearance, measured benchmark PASS and final signed manifest remain pending. See README.md for the handoff.
