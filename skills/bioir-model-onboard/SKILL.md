---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
name: model-onboard
description: Implement an upstream pretrained model natively in BioIR using its optimized components, with checkpoint conversion and numerical validation. Use when adding a new model or architecture to BioIR; includes data pipeline integration when requested.
license: Apache-2.0
metadata:
  author: Tien Pham
---

# Model Onboarding

**Input:** upstream source and revision, a pretrained checkpoint, a writable
BioIR checkout, and the requested model features. **Output:** a native BioIR
implementation, checkpoint conversion, tests and validation evidence, and
data pipeline integration when in scope.

When the request is an offline decision about a recorded input, write
that decision and stop. Do not clone, implement, or run a model.

The procedure lives in the BioIR checkout, written `$BIOIR` below. Read
`$BIOIR/docs/advanced/port-model.md` in full and follow it. Without a
checkout, clone `https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime`
and work on a branch. The guides and the `bench-perf-oss` skill named below
are in that checkout, so stop and report if you can reach neither a checkout
nor the clone. Reuse BioIR's modules where their math matches
upstream, and implement the architecture's differences. An adapter that
only swaps one module into the upstream model does not complete a native
model port; for that task, follow
`$BIOIR/docs/advanced/accelerate-custom-model.md` instead.

## Establish the Scope

Use the source, checkpoint, checkout, and working directory already supplied
in the conversation. Ask only for missing information that affects the port:
the authoritative upstream revision and checkpoint, the model variant, and
whether full data pipeline integration is required.

Work on a branch in the BioIR checkout. Keep upstream read-only and its
dependencies in a separate environment. Store captures, reference outputs,
validation scripts, and reports in an external working directory; keep
reusable implementation and tests in BioIR.

Before implementing, write a feasibility report in the working directory.
Map the upstream components to BioIR components, explain mathematical and
layout differences, list missing operations and input types, and state the
implementation and validation scope. Resolve an ambiguous architecture or
checkpoint with the user; routine implementation choices can proceed.

## Execution and Evidence

- Inspect the available hardware before scheduling validation. Without a
  supported GPU, continue source analysis, implementation, and checks that
  run on CPU. Save the GPU validation commands and mark those checks pending;
  never report the port as validated from static checks alone.
- Load checkpoints with `weights_only=True`. Account for checkpoint
  parameters and buffers: fail on a tensor that is unread or read twice, and
  explain intentionally unused tensors and shared weights. Never hide
  missing mappings with non-strict loading.
- Establish module equivalence on identical captured inputs before relying
  on end-to-end quality scores, then compare the whole model with the
  upstream's spread between repeated runs. Record upstream and BioIR
  revisions, checkpoint identity, settings, tolerances, commands, and
  artifacts.
- When full pipeline support is requested, follow
  `$BIOIR/docs/advanced/port-data-pipeline.md` and
  `$BIOIR/docs/advanced/validate-data-pipeline.md`. Reuse an existing
  pipeline only after checking its feature and output semantics against
  upstream.
- Benchmark only after correctness checks pass. For supported folding
  models, use the `bench-perf-oss` skill at
  `$BIOIR/skills/bench-perf-oss/SKILL.md`; retain its scope and
  applicability rules.

## Report

State what was implemented, which BioIR components were reused or extended,
how checkpoint tensors were accounted for, and how to construct and run the
model. List every check as PASS, FAIL, or PENDING with the command, artifact
path, and sample scope. Give the exit code for executed checks and the reason
for pending checks. Include documented differences, unsupported features,
pending GPU work, and the working directory.
