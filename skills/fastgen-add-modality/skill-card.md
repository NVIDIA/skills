## Description: <br>
Extend NVIDIA FastGen to a new generated or conditioning modality beyond the existing image, video, audio, camera and text paths: representation, loader keys, model state, noise-schedule clock, losses, sampling, serialization and validation. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and ML engineers use this skill to extend NVIDIA FastGen with a new generated or conditioning signal type that must cross the data, model and output interfaces, covering representation, data loading, noise-schedule handling, method support, native output serialization and validation. <br>

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
- [Modality boundaries beyond the network](references/modality-contract.md) <br>
- [FastGen networks README](https://github.com/NVlabs/FastGen/blob/main/fastgen/networks/README.md) <br>
- [FastGen datasets README](https://github.com/NVlabs/FastGen/blob/main/fastgen/datasets/README.md) <br>
- [fastgen-develop skill](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-develop/SKILL.md) <br>
- [fastgen-runtime skill](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-runtime/SKILL.md) <br>
- [fastgen-add-network skill](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-add-network/SKILL.md) <br>
- [fastgen-dataset skill](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-dataset/SKILL.md) <br>
- [fastgen-configure skill](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-configure/SKILL.md) <br>
- [fastgen-validate skill](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-validate/SKILL.md) <br>
- [fastgen-train skill](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-train/SKILL.md) <br>
- [FastGen documentation](https://docs.nvidia.com/fastgen) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions, Shell commands, Analysis] <br>
**Output Format:** [Markdown with inline code blocks and source/config file edits] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Final report lists supported input and output modes, implementation, data and config paths, native artifacts and evidence per tested combination] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
3 evaluation tasks (3 positive), 1 attempt per task, each attempt run in its own isolated sandbox pod (`k8s-sandbox` environment); evaluator version `1.6.8`, evaluation date 2026-11-05. Dataset cases: `adjacent-existing-ltx-audio-video`, `context-independent-clocks`, `explicit-joint-native-export`. <br>

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
| Overall | 84.6% — baseline ran, but no comparable score was available; uplift unavailable | 87.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 66.7% → 100.0% (+33.3 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 61.7% — baseline ran, but no comparable score was available; uplift unavailable | 83.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 90.5% → 92.9% (+2.4 points) | 89.5% → 84.8% (-4.7 points) |
| Efficiency | 68.4% — baseline ran, but no comparable score was available; uplift unavailable | 71.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 81.7% → 80.3% (-1.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 83.9% → 82.0% (-1.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Skill Version(s): <br>
fbf1ad8 (source: git SHA, committed 2026-11-05) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
