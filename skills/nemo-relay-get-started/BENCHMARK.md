# Skill Benchmark: nemo-relay-get-started

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemo-relay-get-started`
- Evaluation date: 2026-10-22
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-sonnet-5-5`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 22 evaluation tasks (21 positive, 1 negative)
- Dataset digest: `sha256:477cc9bd7b45256aa6d8f837938c05e6d79c539ed8bce8bf1908e266f61f3e9f` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 81.0% — baseline ran, but no comparable score was available; uplift unavailable | 71.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 86.4% → 86.4% (±0.0 points) | 79.6% → 70.5% (-9.1 points) |
| Correctness | 33.6% → 84.6% (+51.0 points) | 56.4% → 85.5% (+29.1 points) |
| Discoverability | 92.7% — baseline ran, but no comparable score was available; uplift unavailable | 83.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 34.4% → 62.9% (+28.5 points) | 38.7% → 55.5% (+16.8 points) |
| Efficiency | 78.5% — baseline ran, but no comparable score was available; uplift unavailable | 61.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 8,269,624 | 3,485,549 | +4,784,075 | +137.25% | skill 22/22; base 22/22 |
| claude-code | nemo-relay-get-started-001 | 241,424 | 89,531 | +151,893 | +169.65% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-002 | 465,266 | 89,838 | +375,428 | +417.89% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-003 | 1,463,446 | 153,853 | +1,309,593 | +851.20% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-004 | 89,411 | 29,043 | +60,368 | +207.86% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-005 | 238,131 | 92,203 | +145,928 | +158.27% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-006 | 203,743 | 121,315 | +82,428 | +67.95% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-007 | 471,803 | 89,896 | +381,907 | +424.83% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-008 | 460,852 | 89,464 | +371,388 | +415.13% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-009 | 130,898 | 90,247 | +40,651 | +45.04% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-010 | 152,580 | 133,923 | +18,657 | +13.93% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-011 | 375,286 | 90,170 | +285,116 | +316.20% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-012 | 30,948 | 30,704 | +244 | +0.79% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-013 | 173,818 | 89,630 | +84,188 | +93.93% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-014 | 581,420 | 90,133 | +491,287 | +545.07% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-015 | 392,726 | 125,348 | +267,378 | +213.31% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-016 | 482,696 | 121,978 | +360,718 | +295.72% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-017 | 389,983 | 583,370 | -193,387 | -33.15% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-018 | 615,449 | 770,940 | -155,491 | -20.17% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-019 | 415,012 | 121,441 | +293,571 | +241.74% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-020 | 130,578 | 90,211 | +40,367 | +44.75% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-021 | 297,366 | 237,340 | +60,026 | +25.29% | skill 1/1; base 1/1 |
| claude-code | nemo-relay-get-started-022 | 466,788 | 154,971 | +311,817 | +201.21% | skill 1/1; base 1/1 |
| codex | All cases | 12,363,606 | 8,470,742 | +3,892,864 | +45.96% | skill 22/22; base 22/22 |
| codex | nemo-relay-get-started-001 | 34,536 | 18,302 | +16,234 | +88.70% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-002 | 786,313 | 40,813 | +745,500 | +1826.62% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-003 | 439,518 | 425,422 | +14,096 | +3.31% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-004 | 13,297 | 13,237 | +60 | +0.45% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-005 | 1,054,171 | 625,168 | +429,003 | +68.62% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-006 | 78,490 | 13,436 | +65,054 | +484.18% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-007 | 88,674 | 32,031 | +56,643 | +176.84% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-008 | 39,197 | 74,301 | -35,104 | -47.25% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-009 | 39,604 | 116,973 | -77,369 | -66.14% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-010 | 85,988 | 27,658 | +58,330 | +210.90% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-011 | 32,923 | 51,971 | -19,048 | -36.65% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-012 | 31,298 | 38,018 | -6,720 | -17.68% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-013 | 41,136 | 66,886 | -25,750 | -38.50% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-014 | 54,303 | 39,969 | +14,334 | +35.86% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-015 | 1,781,789 | 552,045 | +1,229,744 | +222.76% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-016 | 4,540,882 | 109,076 | +4,431,806 | +4063.04% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-017 | 90,050 | 31,479 | +58,571 | +186.06% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-018 | 1,220,978 | 5,786,295 | -4,565,317 | -78.90% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-019 | 1,669,588 | 210,266 | +1,459,322 | +694.04% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-020 | 43,549 | 13,420 | +30,129 | +224.51% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-021 | 157,687 | 120,182 | +37,505 | +31.21% | skill 1/1; base 1/1 |
| codex | nemo-relay-get-started-022 | 39,635 | 63,794 | -24,159 | -37.87% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 20,633,230 | 11,956,291 | +8,676,939 | +72.57% | skill 44/44; base 44/44 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 11 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 22 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/nemo-relay-get-started/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in agent-runtimes-try-now.md (`skills/nemo-relay-get-started/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nemo-relay-get-started/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nemo-relay-get-started/SKILL.md`)
- **LOW** QUALITY/quality_correctness: No examples provided (`skills/nemo-relay-get-started/SKILL.md`)
- 6 additional finding(s) are available in the full evaluation artifacts.

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
