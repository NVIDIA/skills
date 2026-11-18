## Description: <br>
Implement an upstream pretrained model natively in BioIR using its optimized components, with checkpoint conversion and numerical validation. Use when adding a new model or architecture to BioIR; includes data pipeline integration when requested. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers working on BioNeMo Inference Runtime (BioIR) use this skill to port an upstream pretrained structure-prediction model natively into BioIR, including a feasibility report, checkpoint conversion with tensor accounting, module-level and whole-model numerical validation, and data pipeline integration when requested. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [BioNeMo Inference Runtime (GitHub)](https://github.com/NVIDIA-BioNeMo/BioNeMo-Inference-Runtime) <br>
- [Port a model (docs/advanced/port-model.md)](../../docs/advanced/port-model.md) <br>
- [Accelerate a custom model (docs/advanced/accelerate-custom-model.md)](../../docs/advanced/accelerate-custom-model.md) <br>
- [Port a data pipeline (docs/advanced/port-data-pipeline.md)](../../docs/advanced/port-data-pipeline.md) <br>
- [Validate a data pipeline (docs/advanced/validate-data-pipeline.md)](../../docs/advanced/validate-data-pipeline.md) <br>
- [bench-perf-oss skill](../bench-perf-oss/SKILL.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Files, Shell commands, Analysis] <br>
**Output Format:** [Python source and tests in the BioIR checkout, plus Markdown feasibility and validation reports listing each check as PASS, FAIL, or PENDING] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Validation artifacts, captures, and reports are kept in an external working directory; GPU checks are marked PENDING when no supported GPU is available.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
9 evaluation tasks (7 positive, 2 negative), 1 attempt per task, each in its own isolated sandbox pod (`k8s-sandbox`), evaluator version `1.6.10`. The cases are synthetic, CPU-only port-decision records covering feasibility planning, checkpoint tensor accounting, buffer handling, module parity versus structure scores, fused-kernel shapes, hub registration, benchmark ordering, and two negative cases (a module swap and a pipeline-only change). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from `security` (100%). <br>
- Correctness: Is the answer correct? Scored from `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Scored from `goal_accuracy` (50%) + `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Scored from `skill_efficiency` (50%) + `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Security: unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Skill Execution: whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool Productivity: tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Accuracy: final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Goal Accuracy: whether the user's goal was achieved. <br>
- `behavior_check`: Behavior Check: whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Token Efficiency: actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
> ✅ **Overall verdict: PASS — Recommended for publication**

### Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 87.9% — baseline ran, but no comparable score was available; uplift unavailable | 88.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 95.6% → 95.6% (±0.0 points) | 80.0% → 88.9% (+8.9 points) |
| Discoverability | 55.7% — baseline ran, but no comparable score was available; uplift unavailable | 68.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 92.6% → 92.6% (±0.0 points) | 80.6% → 86.1% (+5.5 points) |
| Efficiency | 95.5% — baseline ran, but no comparable score was available; uplift unavailable | 96.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.5% → 98.1% (-0.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 93.1% → 92.6% (-0.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 13 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 9 task(s) |

## Skill Version(s): <br>
b088bb87 (source: git SHA, committed 2026-11-18) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
