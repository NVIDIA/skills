---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
{}
---

# Benchmark workflow evaluations

This suite evaluates preparation, diagnosis, and artifact analysis on a CPU.
It does not measure folding accuracy, GPU speed, CUDA graph behavior, or
kernel correctness. Every supplied result and preflight snapshot is synthetic.
The tiny CIF fragments test path staging only and must never be scored as
ground truth. The manifest task explicitly substitutes this preparation fixture
for the production dataset; ordinary benchmark runs still use the release specs.

## Cases

| Case | Task                                             | Failure the case should expose                                                                |
| ---- | ------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| 001  | Prepare a manifest and per-chain staging tree    | Lost chain copies, extra MSAs, templates, or RNA residues                                     |
| 002  | Audit featurized inputs before timing            | Accepting a nonzero mask despite missing template slots or unequal paired rows                |
| 003  | Review a Protenix-v2 runtime lock                | Off-by-one recycle count, wrong feature path, CUDA wheel, compile arm, or weight fallback     |
| 004  | Triage a Boltz-2 quality gap                     | Blaming the engine or destroying BioIR deletion features to hide an OSS discrepancy           |
| 005  | Audit result artifacts                           | Publishing recaptured timings, monomer DockQ zero, mismatched templates, or empty telemetry   |
| 006  | Aggregate supplied latency fixtures              | Ratio inversion, arithmetic mean, wrong bin boundaries, or OOM/failed rows in the denominator |
| 007  | Review checkpoint compatibility                  | Comparing different variants or bypassing a failed strict load                                |
| 008  | Analyze affinity regression predictions          | Activating a structure-folding workflow for a different task                                  |
| 009  | Summarize Ray replica throughput                 | Replacing throughput analysis with serial folding latency                                     |
| 010  | Review an AlphaFold2 multimer plan               | Applying AF3 recycle semantics, allowing early stop, or selecting unsupported polymers        |
| 011  | Audit an approximate-mode plan and paired scores | Omitting the exact control, mislabeling approximate results, or reversing fidelity deltas     |

Case 001 names the skill explicitly. Cases 002–007 and 010–011 describe
realistic tasks without naming it. Cases 008–009 set `expected_skill: null` and
still require useful answers to the actual user request. The negative cases
should not pass merely because the agent refuses to help.

Prompts specify the requested outputs and scope without supplying the acceptance
answers. Assertions describe observable outcomes rather than a required phrase,
command spelling, or list of references to read. The standard evaluator judges
routing, reasoning, tool use, and the final response. The custom grader checks
numeric artifacts, required decisions, staging links, and unchanged input
hashes; it never executes generated agent code.

## Execution and grading

Use SkillEvaluator 1.6.9, matching the onboarding CI contract. That version
requires `assertions`, even though the Anthropic schema example calls the field
`expectations`. Each case has an explicit `files` list. The evaluator mounts
only that case's fixtures under `/workspace/input/` and excludes this `evals/`
directory from agent-visible skill content. The grader and `domain_checks`
reference data are staged under verifier-only `/tests/`, equally for both arms.

No GPU, model weights, API keys, external services, package installation, custom
Dockerfile, or full repository copy is required by these tasks. Keep the
isolated workspace and paired with-skill/without-skill runs. The config requests
`harbor.n_concurrent: 2`, which CI caps against its Harbor concurrency policy.
This is distinct from a global `harbor.case_concurrency` limit; it does not
promise only two active cases across all agents and arms. CI selects agents,
models, attempt count, and its threshold.

`aces_plus_custom` preserves the standard evaluator's score and pass decision.
The `domain_correctness` metric is 1 only when every artifact check and input
hash passes; `domain_check_fraction` helps diagnose failures. Both are
declared in `grading.custom_metrics`; the evaluator binds those names and the
grader source digest when sealing its evidence plan. Both are additional
report metrics, not a replacement CI gate. Review their results
alongside the standard scores before treating a run as evidence of a correct
benchmark workflow. Enforcing those metrics as publication gates would require
an explicit downstream policy change.

For the manifest case, the artifact score checks the final staging links,
manifest, index, and mapper presence. It does not prove that the mapper ran
twice, was silent on its second run, or produced those files itself. The
standard evaluator's assertions cover execution; inspect the agent trajectory
to verify both invocations and their output. A passing artifact score alone
does not establish repeatability. The custom grader never executes the
agent-authored mapper.

Validate structure and task contracts before running paid agent trials:

```bash
skillevaluator tier3 validate skills/bench-perf-oss --strict --harbor-contract
```

The optional evaluator integration test in
`tests/skills/test_bench_perf_oss_evals.py` stages both arms and seals the same
pre-execution plan as CI. Run it with the CI-pinned SkillEvaluator installed;
it skips when that integration is unavailable. This test reproduces the
missing-metric declaration failure before checking the corrected plan.
Schema validation alone does not exercise plan sealing.

After live evaluation, inspect both agents and both arms, invalid/timeout
counts, each case's assertions, domain checks, and trajectories. Keep failures
caused by the environment distinct from behavioral failures. Confirm that the
skill adds value; do not weaken assertions or remove failing cases to produce a
positive lift. Repeat trials when a result is sensitive to model variance.
CLI/config validation and the CPU verifier controls do not establish live-agent
success.

A separate GPU suite would need pinned models, compatible local checkpoints,
dataset digests, scorer environments, and dedicated hardware. Only that suite
could establish actual timing, attachment on real featurized tensors,
compilation stability, and folding-quality behavior. Do not infer those from
this CPU suite.

## Design references

- [Anthropic skill-creator workflow](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md)
- [Anthropic eval JSON schema](https://github.com/anthropics/skills/blob/main/skills/skill-creator/references/schemas.md)
- [Anthropic agent evaluation guidance](https://www.anthropic.com/research/demystifying-evals-for-ai-agents)
