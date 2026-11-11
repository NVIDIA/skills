## Description: <br>
Run, resume or troubleshoot NVIDIA FastGen training: SFT or distillation with train.py, bounded smoke tests, distributed launches, checkpoints, validation and W&B evidence. Use it whenever training is executed or a training run misbehaves. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and ML researchers use this skill to run, resume or troubleshoot FastGen SFT and distillation training: bounding smoke tests and experiments, distinguishing weights, warm starts and full resume, launching single- or multi-GPU (DDP/FSDP) runs, monitoring losses, gradients and samples, and reporting run URLs, resolved config, executed iterations, checkpoints and validation evidence. <br>

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
- [Training, logging and resume contracts](references/run-contract.md) <br>
- [FastGen README quick start](https://github.com/NVlabs/FastGen/blob/main/README.md#quick-start) <br>
- [FastGen configuration README](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/README.md) <br>
- [Distributed checkpoint storage](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/README.md#distributed-checkpoint-storage) <br>
- [FastGen troubleshooting guide](https://github.com/NVlabs/FastGen/blob/main/fern/docs/troubleshooting.mdx) <br>
- [FastGen callbacks README (W&B callback)](https://github.com/NVlabs/FastGen/blob/main/fastgen/callbacks/README.md) <br>
- [fastgen-runtime skill](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-runtime/SKILL.md) <br>
- [fastgen-configure skill](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-configure/SKILL.md) <br>
- [fastgen-dataset skill](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-dataset/SKILL.md) <br>
- [fastgen-validate skill](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-validate/SKILL.md) <br>
- [fastgen-inference skill](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-inference/SKILL.md) <br>
- [FastGen documentation](https://docs.nvidia.com/fastgen) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks and a training run report] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (4 positive), 1 attempt per task, each run in an isolated sandbox pod (`k8s-sandbox`); each task was also attempted without the skill as a baseline. <br>

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

**Publication Recommendation:** Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 88.2% — baseline ran, but no comparable score was available; uplift unavailable | Not available |
| Security | 75.0% → 75.0% (±0.0 points) | Not available |
| Correctness | 100.0% → 100.0% (±0.0 points) | Not available |
| Discoverability | 92.5% — baseline ran, but no comparable score was available; uplift unavailable | Not available |
| Effectiveness | 90.8% → 93.9% (+3.1 points) | Not available |
| Efficiency | 79.3% — baseline ran, but no comparable score was available; uplift unavailable | Not available |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 89.6% → 89.1% (-0.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 16 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Skill Version(s): <br>
fbf1ad8 (source: git SHA, committed 2026-11-05) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
