# Skill Benchmark: cdf-augmentation

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cdf-augmentation`
- Evaluation date: 2026-10-13
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 14 evaluation tasks (13 positive, 1 negative)
- Dataset digest: `sha256:187105966081ef5d05de4b8e34cbd0f0432d018836671dde0564eb960fa31b46` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 90.2% — baseline ran, but no comparable score was available; uplift unavailable | 83.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 92.9% → 92.9% (±0.0 points) | 78.6% → 92.9% (+14.3 points) |
| Correctness | 44.3% → 98.6% (+54.3 points) | 52.9% → 84.3% (+31.4 points) |
| Discoverability | 87.3% — baseline ran, but no comparable score was available; uplift unavailable | 71.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 37.1% → 83.5% (+46.4 points) | 45.3% → 77.2% (+31.9 points) |
| Efficiency | 88.7% — baseline ran, but no comparable score was available; uplift unavailable | 92.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,593,118 | 4,587,421 | -994,303 | -21.67% | skill 14/14; base 14/14 |
| claude-code | aug-attribute-verification-setup-positive | 126,787 | 391,712 | -264,925 | -67.63% | skill 1/1; base 1/1 |
| claude-code | aug-batch-config-conditional-vars-positive | 376,307 | 185,219 | +191,088 | +103.17% | skill 1/1; base 1/1 |
| claude-code | aug-captioning-attribute-clothing-positive | 187,020 | 377,646 | -190,626 | -50.48% | skill 1/1; base 1/1 |
| claude-code | aug-config-error-missing-cosmos-endpoint-positive | 119,873 | 304,873 | -185,000 | -60.68% | skill 1/1; base 1/1 |
| claude-code | aug-cosmos3-wsm-upload-config-positive | 262,481 | 452,405 | -189,924 | -41.98% | skill 1/1; base 1/1 |
| claude-code | aug-defect-alignment-positive | 189,666 | 236,915 | -47,249 | -19.94% | skill 1/1; base 1/1 |
| claude-code | aug-docker-launch-recipe-positive | 349,827 | 151,679 | +198,148 | +130.64% | skill 1/1; base 1/1 |
| claude-code | aug-image-edit-single-image-positive | 124,807 | 91,298 | +33,509 | +36.70% | skill 1/1; base 1/1 |
| claude-code | aug-model-select-text-to-video-positive | 125,652 | 121,497 | +4,155 | +3.42% | skill 1/1; base 1/1 |
| claude-code | aug-model-select-video-weather-positive | 121,512 | 349,584 | -228,072 | -65.24% | skill 1/1; base 1/1 |
| claude-code | aug-no-hardcoded-keys-security | 29,964 | 153,166 | -123,202 | -80.44% | skill 1/1; base 1/1 |
| claude-code | aug-no-secret-leak-security | 59,191 | 29,463 | +29,728 | +100.90% | skill 1/1; base 1/1 |
| claude-code | aug-reject-training-request-negative | 67,949 | 120,765 | -52,816 | -43.73% | skill 1/1; base 1/1 |
| claude-code | aug-unrelated-request-negative | 1,452,082 | 1,621,199 | -169,117 | -10.43% | skill 1/1; base 1/1 |
| codex | All cases | 1,929,642 | 2,058,776 | -129,134 | -6.27% | skill 14/14; base 14/14 |
| codex | aug-attribute-verification-setup-positive | 231,033 | 55,453 | +175,580 | +316.63% | skill 1/1; base 1/1 |
| codex | aug-batch-config-conditional-vars-positive | 139,586 | 69,817 | +69,769 | +99.93% | skill 1/1; base 1/1 |
| codex | aug-captioning-attribute-clothing-positive | 147,623 | 423,557 | -275,934 | -65.15% | skill 1/1; base 1/1 |
| codex | aug-config-error-missing-cosmos-endpoint-positive | 158,020 | 89,878 | +68,142 | +75.82% | skill 1/1; base 1/1 |
| codex | aug-cosmos3-wsm-upload-config-positive | 198,456 | 370,564 | -172,108 | -46.44% | skill 1/1; base 1/1 |
| codex | aug-defect-alignment-positive | 115,388 | 103,889 | +11,499 | +11.07% | skill 1/1; base 1/1 |
| codex | aug-docker-launch-recipe-positive | 83,462 | 99,601 | -16,139 | -16.20% | skill 1/1; base 1/1 |
| codex | aug-image-edit-single-image-positive | 67,529 | 60,448 | +7,081 | +11.71% | skill 1/1; base 1/1 |
| codex | aug-model-select-text-to-video-positive | 92,265 | 80,553 | +11,712 | +14.54% | skill 1/1; base 1/1 |
| codex | aug-model-select-video-weather-positive | 89,352 | 83,627 | +5,725 | +6.85% | skill 1/1; base 1/1 |
| codex | aug-no-hardcoded-keys-security | 13,604 | 83,605 | -70,001 | -83.73% | skill 1/1; base 1/1 |
| codex | aug-no-secret-leak-security | 13,412 | 13,345 | +67 | +0.50% | skill 1/1; base 1/1 |
| codex | aug-reject-training-request-negative | 31,131 | 78,758 | -47,627 | -60.47% | skill 1/1; base 1/1 |
| codex | aug-unrelated-request-negative | 548,781 | 445,681 | +103,100 | +23.13% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 5,522,760 | 6,646,197 | -1,123,437 | -16.90% | skill 28/28; base 28/28 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 21 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 14 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_efficiency: Large skill (5300 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/cdf-augmentation/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in config-decision-tree.md (`skills/cdf-augmentation/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/cdf-augmentation/SKILL.md`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/cdf-augmentation/SKILL.md (`BENCHMARK.md:104`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/cdf-augmentation/SKILL.md (`BENCHMARK.md:105`)
- 16 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
