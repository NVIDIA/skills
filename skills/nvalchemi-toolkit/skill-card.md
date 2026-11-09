## Description: <br>
Build and change NVIDIA ALCHEMI Toolkit (nvalchemi) workflows for atomistic simulation and machine-learned interatomic potentials (MLIPs), including batched structure relaxation, molecular dynamics, reaction paths, MLIP training or fine-tuning, Zarr atomistic datasets, generative structure workflows, model wrapping, and custom hooks or integrators. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and computational scientists use this skill to have an agent write or change Python code that uses nvalchemi for atomic data and Zarr storage, MLIP model wrappers, batched relaxation and MD, reaction paths and enhanced sampling, training, fine-tuning, distillation, generative sampling, and custom hooks or integrators. It is not for other simulation packages without nvalchemi, ALCHEMI Toolkit-Ops (nvalchemiops) kernel work, or plain PyTorch questions. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [No] <br>
**Credential Type(s):** [None] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [references/data.md](references/data.md) <br>
- [references/models.md](references/models.md) <br>
- [references/dynamics-optimization.md](references/dynamics-optimization.md) <br>
- [references/training.md](references/training.md) <br>
- [references/generative-workflows.md](references/generative-workflows.md) <br>
- [references/composition-and-extension.md](references/composition-and-extension.md) <br>
- [NVIDIA ALCHEMI Toolkit documentation](https://nvidia.github.io/nvalchemi-toolkit/) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Analysis] <br>
**Output Format:** [Markdown with inline Python and bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Exploratory runs are capped at 3 systems, 100 steps, and under one minute of wall time; measured results are reported with units only for executed work, otherwise marked "Not executed".] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
9 evaluation tasks (7 positive, 2 negative), 1 attempt per task, each in its own isolated sandbox pod (`k8s-sandbox` environment), with no-skill baseline runs for comparison; evaluator version `1.6.9`, evaluated 2026-11-07. <br>

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
✅ **Overall verdict: PASS — Recommended for publication**

### Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 93.1% — baseline ran, but no comparable score was available; uplift unavailable | 88.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 94.4% → 100.0% (+5.6 points) | 94.4% → 100.0% (+5.6 points) |
| Correctness | 88.9% → 95.6% (+6.7 points) | 82.2% → 97.8% (+15.6 points) |
| Discoverability | 95.0% — baseline ran, but no comparable score was available; uplift unavailable | 82.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 76.7% → 87.3% (+10.6 points) | 70.8% → 87.0% (+16.2 points) |
| Efficiency | 87.6% — baseline ran, but no comparable score was available; uplift unavailable | 76.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 80.9% → 85.3% (+4.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 81.3% → 84.1% (+2.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 6 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 9 task(s) |

## Skill Version(s): <br>
53f09be (source: git SHA, committed 2026-11-06); skill content targets nvalchemi release 0.3.0 (source: SKILL.md) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
