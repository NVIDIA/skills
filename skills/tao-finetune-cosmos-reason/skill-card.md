## Description: <br>
Shared Cosmos3 frontend that explicitly routes Cosmos Framework and Cosmos-RL, validates runtime model/video-dataset/SLURM inputs, consumes an SQSH or packaged backend image, optionally plans explicit clean source builds, prepares checkpoints, validates the first update in-process, and returns token-weighted losses and task-aware accuracy. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and ML engineers use this skill to fine-tune, evaluate, export, and run inference on Cosmos3 (Nano and Edge) vision-language models over video datasets with the Cosmos Framework or Cosmos-RL backends, on Docker or SLURM platforms. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [detailed-guide.md](references/detailed-guide.md) <br>
- [cosmos-reason-launch.md](references/cosmos-reason-launch.md) <br>
- [cosmos-reason-evaluate.md](references/cosmos-reason-evaluate.md) <br>
- [cosmos-reason-parameters.md](references/cosmos-reason-parameters.md) <br>
- [cosmos-actions-parameters.md](references/cosmos-actions-parameters.md) <br>
- [cosmos-backend-operations.md](references/cosmos-backend-operations.md) <br>
- [cosmos-data-specs.md](references/cosmos-data-specs.md) <br>
- [cosmos-reproducibility-gates.md](references/cosmos-reproducibility-gates.md) <br>
- [cosmos-reason-single-gpu-video.md](references/cosmos-reason-single-gpu-video.md) <br>
- [cosmos-reason-automl.md](references/cosmos-reason-automl.md) <br>
- [cosmos-automl-deft.md](references/cosmos-automl-deft.md) <br>
- [skill_info.yaml](references/skill_info.yaml) <br>
- [TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Files, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks, backend-native TOML/YAML specs, and JSON plan/metric artifacts] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Training/evaluation metrics extracted from job outputs; job state tracked in job records] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
7 evaluation tasks (7 positive), 1 attempt per task, run in the `k8s-sandbox` environment with each task attempt in its own isolated sandbox pod; each task was also attempted without the skill as a baseline. <br>

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
| Overall | 86.4% — baseline ran, but no comparable score was available; uplift unavailable | 75.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 42.9% → 91.4% (+48.5 points) | 57.1% → 82.9% (+25.8 points) |
| Discoverability | 90.0% — baseline ran, but no comparable score was available; uplift unavailable | 62.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 19.1% → 62.0% (+42.9 points) | 17.1% → 39.0% (+21.9 points) |
| Efficiency | 88.7% — baseline ran, but no comparable score was available; uplift unavailable | 95.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.8% → 92.8% (-5.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 86.8% → 90.7% (+3.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 95 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 31 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 7 task(s) |

## Skill Version(s): <br>
0.3.6 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
