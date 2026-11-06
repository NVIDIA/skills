## Description: <br>
Use for NVIDIA-related requests where an NVIDIA skill might help, even if the user did not ask for a skill; it checks the live NVIDIA skills catalog and recommends matching NVIDIA skills (installing only with user approval), while staying quiet for generic non-NVIDIA route, optimize, deploy, AI, video, data, or infrastructure tasks. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers using AI coding agents (such as Claude Code, Codex, or Hermes Agent) use this skill to discover, recommend, and—with explicit approval—install NVIDIA agent skills relevant to NVIDIA products, SDKs, hardware, and workflows such as CUDA, NIM, NeMo, Omniverse/OpenUSD, Isaac, Jetson, RAPIDS, cuOpt, Dynamo, Holoscan, TensorRT, DeepStream, and VSS. <br>

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
- [Hermes Agent: Catalog Lookup and Install](references/hermes-catalog.md) <br>
- [NVIDIA Taxonomy Routing](references/taxonomy-routing.md) <br>
- [NVIDIA Skills catalog (GitHub)](https://github.com/NVIDIA/skills/tree/main/skills) <br>
- [NVIDIA Skills on build.nvidia.com](https://build.nvidia.com/skills) <br>
- [NVIDIA Skills catalog index (skills.sh.json)](https://raw.githubusercontent.com/NVIDIA/skills/main/skills.sh.json) <br>
- [NVIDIA Skills marketplace metadata (metadata.json)](https://raw.githubusercontent.com/NVIDIA/skills/main/.github/scripts/marketplace/metadata.json) <br>
- [NVIDIA Skills documentation](https://docs.nvidia.com/skills) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks (skill recommendations with name, rationale, install command, and first prompt)] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Recommends at most three skills ordered by confidence; installs only after explicit user approval.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
18 evaluation tasks (13 positive, 5 negative); 1 attempt per task, each in its own isolated sandbox pod (`k8s-sandbox` environment), with each task also run as a no-skill baseline. Evaluator version `1.6.8`; evaluation date 2026-11-05. <br>

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
| Overall | 94.0% — baseline ran, but no comparable score was available; uplift unavailable | 89.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 97.2% → 100.0% (+2.8 points) | 100.0% → 94.4% (-5.6 points) |
| Correctness | 61.1% → 94.4% (+33.3 points) | 74.4% → 93.3% (+18.9 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 36.5% → 96.5% (+60.0 points) | 56.5% → 100.0% (+43.5 points) |
| Efficiency | 79.2% — baseline ran, but no comparable score was available; uplift unavailable | 63.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.4% → 93.5% (-4.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 90.3% → 88.3% (-2.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Skill Version(s): <br>
0.3.0 (source: changelog, released 2025-05-21; git SHA f73b929, committed 2026-11-05) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
