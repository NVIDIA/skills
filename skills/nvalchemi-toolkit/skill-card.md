## Description: <br>
Build and change NVIDIA ALCHEMI Toolkit (nvalchemi) workflows for atomistic simulation and machine-learned interatomic potentials (MLIPs), including batched structure relaxation, molecular dynamics, reaction paths, training or fine-tuning an MLIP, atomistic datasets in Zarr, generative structure workflows, model wrapping, and custom hooks or integrators. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>

## Use Case: <br>
Developers, researchers, and engineers use this skill to have a coding agent write, change, and check Python code that uses the nvalchemi package for batched GPU atomistic simulation, MLIP training and fine-tuning, atomistic data handling, and custom toolkit extensions. It is not for ASE-only, OpenMM, LAMMPS, or GROMACS tasks that do not involve nvalchemi, nvalchemiops kernel work, or plain PyTorch questions. <br>

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
- [Data note (references/data.md)](references/data.md) <br>
- [Models note (references/models.md)](references/models.md) <br>
- [Dynamics and optimization note (references/dynamics-optimization.md)](references/dynamics-optimization.md) <br>
- [Training note (references/training.md)](references/training.md) <br>
- [Generative workflows note (references/generative-workflows.md)](references/generative-workflows.md) <br>
- [Composition and extension note (references/composition-and-extension.md)](references/composition-and-extension.md) <br>
- [NVIDIA ALCHEMI Toolkit documentation](https://nvidia.github.io/nvalchemi-toolkit/) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Analysis] <br>
**Output Format:** [Markdown with inline Python and bash code blocks; Python scripts written to the working folder] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Exploratory runs are capped at 3 systems, 100 steps, and under one minute of wall time; measured results are reported with units only for work that ran, and unexecuted work is marked "Not executed".] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
9 evaluation tasks (7 positive, 2 negative), 1 attempt per task, each run in its own isolated sandbox pod (`k8s-sandbox` environment). Each task was also attempted without the skill as a baseline. Evaluator version `1.6.9`, evaluation date 2026-11-14. <br>

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
**Overall verdict: PASS — Recommended for publication**

**Publication Recommendation:** Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 91.1% — baseline ran, but no comparable score was available; uplift unavailable | 87.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 91.1% → 97.8% (+6.7 points) | 91.1% → 93.3% (+2.2 points) |
| Discoverability | 85.0% — baseline ran, but no comparable score was available; uplift unavailable | 85.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 83.2% → 89.2% (+6.0 points) | 75.3% → 83.4% (+8.1 points) |
| Efficiency | 83.7% — baseline ran, but no comparable score was available; uplift unavailable | 75.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 82.1% → 85.8% (+3.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 82.4% → 84.5% (+2.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 6 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 9 task(s) |

## Skill Version(s): <br>
0f44153 (source: git SHA, committed 2026-11-14); notes target the nvalchemi 0.3.0 release API (source: SKILL.md) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
