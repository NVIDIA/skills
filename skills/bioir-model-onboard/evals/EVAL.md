---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
{}
---

# Model-onboarding evaluations

This suite evaluates port decisions on a CPU. It does not implement a model,
load a checkpoint, or measure numerical or GPU behavior. Every fixture is a
synthetic decision record.

## Cases

- 001 plans a Protenix-v2 fork when the revision, checkpoint, and GPU are
  absent.
- 002 fails a conversion that leaves a tensor unread and reads another twice.
- 003 rejects a silent skip of recomputed distance-bin buffers.
- 004 refuses to treat a close lDDT as module parity.
- 005 keeps an unsupported fused-kernel width on the fallback path.
- 006 registers a compute-only model in the hub without claiming
  `build_processor`.
- 007 answers a Pairformer swap without a native model port.
- 008 holds a folding speedup until correctness checks pass.
- 009 answers a pipeline-only change to an existing native model.

Case 001 names the skill. Cases 002–006 and 008 describe the task without
naming it. Cases 007 and 009 set `expected_skill: null` and still require a
useful answer. A negative case should not pass merely because the agent
refuses.

Prompts name the output fields and allowed values without giving the
acceptance answer. Assertions describe observable outcomes. The custom grader
checks the decision artifact and unchanged input hashes. It never executes
generated agent code.

## Execution and grading

Use SkillEvaluator 1.6.9, matching the onboarding CI contract. Each case has
`assertions` and an explicit `files` list. The evaluator mounts only that
case's fixture under `/workspace/input/` and excludes this `evals/` directory
from agent-visible skill content. The grader and `domain_checks` reference
data are staged under verifier-only `/tests/`.

No GPU, model weights, API keys, external services, package installation, or
full repository copy is required. Keep the isolated workspace and paired
with-skill and without-skill runs. The config requests
`harbor.n_concurrent: 1`, which CI caps against its Harbor concurrency
policy. This is distinct from a global `harbor.case_concurrency` limit; it
does not promise only one active case across all agents and arms. CI selects
agents, models, attempt count, and its threshold.

`aces_plus_custom` preserves the standard evaluator's score and pass
decision. `domain_correctness` is 1 only when every artifact check and input
hash passes. `domain_check_fraction` helps diagnose failures. Both are
additional report metrics, not a replacement CI gate.

Validate structure and task contracts before running paid agent trials:

```bash
skillevaluator tier3 validate skills/model-onboard --strict --harbor-contract
```

The optional evaluator integration test in
`tests/skills/test_onboarded_skill_evals.py` stages both arms and seals the
same pre-execution plan as CI. It skips when that integration is unavailable.
Schema validation alone does not exercise plan sealing.

CLI validation and these CPU checks do not establish that a live agent
completed a real model port. Do not weaken assertions or remove failing
cases to produce a positive lift.
