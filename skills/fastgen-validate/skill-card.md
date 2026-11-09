## Description: <br>
Validate NVIDIA FastGen integrations and training quality: numerical parity against reference implementations, seeded side-by-side media, gradient and trainability checks, checkpoint round trips and repository tests. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and ML engineers working with NVIDIA FastGen use this skill whenever a claim about correctness, parity or quality needs evidence, producing manifests, parity comparisons, side-by-side media, trainability checks, distillation-quality measurements and repository test runs with explicit pass/fail/not-run states. <br>

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
- [fastgen-runtime skill](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-runtime/SKILL.md) <br>
- [fastgen-inference skill](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-inference/SKILL.md) <br>
- [fastgen-train skill](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-train/SKILL.md) <br>
- [New-network validation checklist](https://github.com/NVlabs/FastGen/blob/main/fastgen/networks/README.md#validate-a-new-network) <br>
- [Run contract: interpret media and scalars](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-train/references/run-contract.md#interpret-media-and-scalars-correctly) <br>
- [Run contract: resume restores state](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-train/references/run-contract.md#resume-restores-state-not-the-exact-trajectory) <br>
- [FID evaluation script](https://github.com/NVlabs/FastGen/blob/main/scripts/README.md#fid-evaluation) <br>
- [FastGen tests](https://github.com/NVlabs/FastGen/tree/main/tests) <br>
- [CONTRIBUTING.md: continuous integration](https://github.com/NVlabs/FastGen/blob/main/CONTRIBUTING.md#continuous-integration) <br>
- [FastGen documentation](https://docs.nvidia.com/fastgen) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Code, Shell commands, Files] <br>
**Output Format:** [Markdown validation report with commands, manifests, metrics and pass/fail/not-run states; test code and JSON artifacts] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (4 positive), 1 attempt per task, each attempt run in its own isolated sandbox pod (`k8s-sandbox` environment); evaluator version `1.6.8`, evaluation date 2026-11-06. <br>

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
**Overall verdict:** ✅ **PASS — Recommended for publication**

**Publication Recommendation:** Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 88.6% — baseline ran, but no comparable score was available; uplift unavailable | Not available |
| Security | 100.0% → 75.0% (-25.0 points) | Not available |
| Correctness | 100.0% → 100.0% (±0.0 points) | Not available |
| Discoverability | 81.3% — baseline ran, but no comparable score was available; uplift unavailable | Not available |
| Effectiveness | 97.5% → 100.0% (+2.5 points) | Not available |
| Efficiency | 86.8% — baseline ran, but no comparable score was available; uplift unavailable | Not available |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 92.6% → 92.8% (+0.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 13 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Skill Version(s): <br>
fbf1ad8 (source: git SHA, committed 2026-11-05) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
