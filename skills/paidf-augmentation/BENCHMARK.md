# Skill Benchmark: paidf-augmentation

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `paidf-augmentation`
- Evaluation date: 2026-09-17
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 14 evaluation tasks (13 positive, 1 negative)
- Dataset digest: `sha256:35b95353d3588e053b4d8c84f9d6067e5d33e13a9789e5048e8817daac2cb64b` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 3
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
| Overall | 92.6% — baseline ran, but no comparable score was available; uplift unavailable | 81.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 96.6% → 100.0% (+3.4 points) | 80.4% → 93.3% (+12.9 points) |
| Correctness | 24.8% → 95.7% (+70.9 points) | 45.7% → 78.7% (+33.0 points) |
| Discoverability | 92.3% — baseline ran, but no comparable score was available; uplift unavailable | 71.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 23.2% → 83.4% (+60.2 points) | 27.4% → 77.2% (+49.8 points) |
| Efficiency | 91.6% — baseline ran, but no comparable score was available; uplift unavailable | 84.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,955,978 | 7,791,489 | N/A | N/A | skill 14/14; base 29/29 |
| claude-code | aug-attribute-verification-setup-positive | 127,843 | 553,662 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | aug-batch-config-conditional-vars-positive | 492,999 | 217,532 | +275,467 | +126.63% | skill 1/1; base 1/1 |
| claude-code | aug-captioning-attribute-clothing-positive | 191,298 | 573,843 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | aug-config-error-missing-cosmos-endpoint-positive | 121,008 | 365,192 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | aug-cosmos3-wsm-upload-config-positive | 127,977 | 685,554 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | aug-defect-alignment-positive | 188,462 | 671,103 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | aug-docker-launch-recipe-positive | 451,442 | 583,739 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | aug-image-edit-single-image-positive | 113,433 | 334,926 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | aug-model-select-text-to-video-positive | 126,812 | 338,687 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | aug-model-select-video-weather-positive | 133,858 | 126,264 | +7,594 | +6.01% | skill 1/1; base 1/1 |
| claude-code | aug-no-hardcoded-keys-security | 68,403 | 60,261 | +8,142 | +13.51% | skill 1/1; base 1/1 |
| claude-code | aug-no-secret-leak-security | 29,316 | 29,472 | -156 | -0.53% | skill 1/1; base 1/1 |
| claude-code | aug-reject-training-request-negative | 68,763 | 161,738 | -92,975 | -57.48% | skill 1/1; base 1/1 |
| claude-code | aug-unrelated-request-negative | 714,364 | 3,089,516 | -2,375,152 | -76.88% | skill 1/1; base 1/1 |
| codex | All cases | 2,079,134 | 5,409,074 | N/A | N/A | skill 15/15; base 28/28 |
| codex | aug-attribute-verification-setup-positive | 142,349 | 878,484 | N/A | N/A | skill 1/1; base 3/3 |
| codex | aug-batch-config-conditional-vars-positive | 247,933 | 84,482 | +163,451 | +193.47% | skill 1/1; base 1/1 |
| codex | aug-captioning-attribute-clothing-positive | 173,908 | 1,285,174 | N/A | N/A | skill 1/1; base 3/3 |
| codex | aug-config-error-missing-cosmos-endpoint-positive | 99,486 | 83,550 | +15,936 | +19.07% | skill 1/1; base 1/1 |
| codex | aug-cosmos3-wsm-upload-config-positive | 168,197 | 1,276,724 | N/A | N/A | skill 1/1; base 3/3 |
| codex | aug-defect-alignment-positive | 150,411 | 441,888 | -291,477 | -65.96% | skill 1/1; base 1/1 |
| codex | aug-docker-launch-recipe-positive | 163,715 | 245,640 | N/A | N/A | skill 1/1; base 3/3 |
| codex | aug-image-edit-single-image-positive | 144,641 | 275,216 | N/A | N/A | skill 2/2; base 3/3 |
| codex | aug-model-select-text-to-video-positive | 120,155 | 118,629 | N/A | N/A | skill 1/1; base 3/3 |
| codex | aug-model-select-video-weather-positive | 73,224 | 55,549 | +17,675 | +31.82% | skill 1/1; base 1/1 |
| codex | aug-no-hardcoded-keys-security | 13,626 | 85,253 | -71,627 | -84.02% | skill 1/1; base 1/1 |
| codex | aug-no-secret-leak-security | 13,454 | 13,359 | +95 | +0.71% | skill 1/1; base 1/1 |
| codex | aug-reject-training-request-negative | 68,297 | 304,327 | N/A | N/A | skill 1/1; base 3/3 |
| codex | aug-unrelated-request-negative | 499,738 | 260,799 | +238,939 | +91.62% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 5,035,112 | 13,200,563 | N/A | N/A | skill 29/29; base 57/57 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 18 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 14 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_efficiency: Large skill (5309 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/paidf-augmentation/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in config-decision-tree.md (`skills/paidf-augmentation/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/paidf-augmentation/SKILL.md`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/paidf-augmentation/SKILL.md (`BENCHMARK.md:104`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/paidf-augmentation/SKILL.md (`BENCHMARK.md:105`)
- 13 additional finding(s) are available in the full evaluation artifacts.

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
