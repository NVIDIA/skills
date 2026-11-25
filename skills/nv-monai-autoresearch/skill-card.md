## Description: <br>
Used for NV-MONAI-Autoresearch medical-imaging research, experiment planning, diagnosis, and resumable optimization. Not for clinical decisions or standalone model inference. <br>

This skill is for research and development only. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers, engineers, and researchers use this skill to run the MIAgent v3.5 medical-imaging research workflow (understanding, readiness, research, candidate selection, implementation, internal/external experiment loops, and freeze) with a durable, human-inspectable record in a caller-owned workspace. Engineering and research use only; not for clinical decisions. <br>

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
- [Critic review](references/critic-review.md) <br>
- [Result analysis](references/result-analysis.md) <br>
- [Segmentation analysis](references/segmentation-analysis.md) <br>
- [Registration analysis](references/registration-analysis.md) <br>
- [Generation analysis](references/generation-analysis.md) <br>
- [Representation analysis](references/representation-analysis.md) <br>
- [VLM analysis](references/vlm-analysis.md) <br>


## Skill Output: <br>
**Output Type(s):** [Files, Analysis, Shell commands] <br>
**Output Format:** [Markdown research records plus JSON initialization and checker reports] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Initialization creates only the challenge, dashboard, and trajectory records and runs no experiments; run artifacts are written to the caller's workspace, never into the installed skill.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
8 evaluation tasks (6 positive, 2 negative), 1 attempt per task, each run in its own isolated sandbox pod (`k8s-sandbox` environment); dataset digest `sha256:09f4cf8c4007172a5d80be1b2be449dd58cb53ed812787cb0d7b8d775f638c98` (skill-evaluator-dataset-snapshot/1); evaluator version `1.6.10`; evaluation date 2026-11-25. Each task was also attempted without the skill as a baseline. Tasks: execution-failure-is-not-result, initialize-only, negative-clinical-decision, negative-single-inference, plateau-external-loop, protected-label-proposal, research-before-method, resume-existing-authority. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from `security` (100%). <br>
- Correctness: Is the answer correct? Scored from `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` (50%) and `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? `skill_efficiency` (50%) + `token_efficiency` (50%); N/A sources are omitted, remaining weights renormalized, and the dimension marked partial. <br>
- Thresholds: Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%. Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between are NEUTRAL. Overall verdict is PASS only when every configured dimension passes for at least one supported agent; lift is diagnostic and does not override this gate. The 50% attempt pass threshold is a separate per-task gate. <br>

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
| Overall | 92.1% — baseline ran, but no comparable score was available; uplift unavailable | 90.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 87.5% → 100.0% (+12.5 points) |
| Correctness | 62.5% → 97.5% (+35.0 points) | 57.5% → 97.5% (+40.0 points) |
| Discoverability | 96.7% — baseline ran, but no comparable score was available; uplift unavailable | 89.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 59.9% → 70.3% (+10.4 points) | 59.9% → 72.9% (+13.0 points) |
| Efficiency | 95.9% — baseline ran, but no comparable score was available; uplift unavailable | 91.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.0% → 94.6% (-3.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 91.0% → 89.2% (-1.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 18 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
