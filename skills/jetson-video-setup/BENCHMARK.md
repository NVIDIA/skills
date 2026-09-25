# Skill Benchmark: jetson-video-setup

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `jetson-video-setup`
- Evaluation date: 2026-09-16
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 5 evaluation tasks (5 positive)
- Dataset digest: `sha256:b449a2c4d6415eebdcc29fc19f769c7af41d2dd4599b0a624951191e5618b55a` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 83.0% — baseline ran, but no comparable score was available; uplift unavailable | 83.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 72.2% → 100.0% (+27.8 points) | 56.7% → 100.0% (+43.3 points) |
| Correctness | 35.6% → 80.0% (+44.4 points) | 50.7% → 88.0% (+37.3 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 87.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 33.8% → 55.2% (+21.4 points) | 23.6% → 61.2% (+37.6 points) |
| Efficiency | 79.9% — baseline ran, but no comparable score was available; uplift unavailable | 80.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 983,484 | 3,691,954 | N/A | N/A | skill 5/5; base 9/9 |
| claude-code | setup-ambiguous-surface-clarification | 95,950 | 770,954 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | setup-device-memory-not-interoperability | 62,818 | 121,843 | -59,025 | -48.44% | skill 1/1; base 1/1 |
| claude-code | setup-native-only-smoke | 276,209 | 1,795,380 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | setup-pynvc-only-smoke | 282,288 | 720,756 | -438,468 | -60.83% | skill 1/1; base 1/1 |
| claude-code | setup-reuse-pynvc-smoke | 266,219 | 283,021 | -16,802 | -5.94% | skill 1/1; base 1/1 |
| codex | All cases | 455,049 | 5,911,838 | N/A | N/A | skill 5/5; base 15/15 |
| codex | setup-ambiguous-surface-clarification | 29,181 | 403,880 | N/A | N/A | skill 1/1; base 3/3 |
| codex | setup-device-memory-not-interoperability | 29,375 | 236,161 | N/A | N/A | skill 1/1; base 3/3 |
| codex | setup-native-only-smoke | 136,949 | 3,096,383 | N/A | N/A | skill 1/1; base 3/3 |
| codex | setup-pynvc-only-smoke | 138,472 | 1,681,304 | N/A | N/A | skill 1/1; base 3/3 |
| codex | setup-reuse-pynvc-smoke | 121,072 | 494,110 | N/A | N/A | skill 1/1; base 3/3 |
| ALL AGENTS | Dataset aggregate | 1,438,533 | 9,603,792 | N/A | N/A | skill 10/10; base 24/24 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 15 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in setup-install.md (`skills/jetson-video-setup/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/jetson-video-setup/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/jetson-video-setup/SKILL.md`)
- **MEDIUM** SECURITY/Sudo/Root Execution (PE2): Privilege Escalation: sudo  (`SKILL.md:166`)
- **MEDIUM** SECURITY/Sudo/Root Execution (PE2): Privilege Escalation: sudo  (`references/setup-install.md:36`)
- 10 additional finding(s) are available in the full evaluation artifacts.

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
