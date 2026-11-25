## Description: <br>
Changes DOCA versions safely — upgrading or downgrading the host DOCA packages, refreshing the BlueField BFB, bumping the NGC DOCA container tag, or rolling back — with the discipline detect -> report -> ASK -> only then a guided upgrade, never upgrading automatically. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers operating NVIDIA DOCA hosts, BlueField DPUs, ConnectX NICs, or NGC DOCA containers use this skill to detect the installed DOCA release, report the gap to a newer target, obtain explicit confirmation before any guided upgrade, downgrade, or rollback, and recover failed or partial upgrades. <br>

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
- [CAPABILITIES.md (upgrade-mode taxonomy, never-auto rule, error taxonomy, safety overlay)](CAPABILITIES.md) <br>
- [TASKS.md (detect, confirmation-gated guided upgrade, verification, failed-upgrade diagnosis)](TASKS.md) <br>
- [DOCA Compatibility Policy](https://docs.nvidia.com/doca/sdk/doca-compatibility-policy/index.html) <br>
- [NVIDIA DOCA SDK documentation](https://docs.nvidia.com/doca/sdk/index.html) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Read-only detection commands may be run by the agent; package, BFB, firmware, and reboot commands are handed to the user only after explicit confirmation.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), 1 attempt per task, each attempt in its own isolated sandbox pod (`k8s-sandbox` environment); dataset digest `sha256:e8be7babdc43b3f9a24fa01668d434ce2211db9e76f46235c2fd58c6e060860f` (skill-evaluator-dataset-snapshot/1); evaluator version `1.6.10`; evaluation date 2026-11-23. Baseline is the same task attempted without the target skill. <br>

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
| Overall | 92.6% — baseline ran, but no comparable score was available; uplift unavailable | 90.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 65.0% → 100.0% (+35.0 points) | 80.0% → 100.0% (+20.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 47.9% → 64.6% (+16.7 points) | 64.6% → 81.3% (+16.7 points) |
| Efficiency | 98.4% — baseline ran, but no comparable score was available; uplift unavailable | 73.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.5% → 96.8% (-1.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 87.3% → 89.5% (+2.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 16 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Skill Version(s): <br>
444e267 (source: git SHA, committed 2026-11-23) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
