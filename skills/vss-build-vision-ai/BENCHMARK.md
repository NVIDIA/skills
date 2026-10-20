# Skill Benchmark: vss-build-vision-ai

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `vss-build-vision-ai`
- Evaluation date: 2026-10-20
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 11 evaluation tasks (10 positive, 1 negative)
- Dataset digest: `sha256:726fa1cf2863b647abfdc7459b34909f535f7558068c245123a9a1ce9f1bf28e` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 73.7% — baseline ran, but no comparable score was available; uplift unavailable | 59.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 72.7% → 90.9% (+18.2 points) | 40.9% → 50.0% (+9.1 points) |
| Correctness | 14.6% → 70.9% (+56.3 points) | 38.2% → 70.9% (+32.7 points) |
| Discoverability | 99.8% — baseline ran, but no comparable score was available; uplift unavailable | 80.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 6.7% → 27.5% (+20.8 points) | 14.7% → 21.0% (+6.3 points) |
| Efficiency | 79.5% — baseline ran, but no comparable score was available; uplift unavailable | 74.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 7,570,476 | 7,318,634 | +251,842 | +3.44% | skill 11/11; base 11/11 |
| claude-code | build-alerts-gb300 | 286,418 | 211,605 | +74,813 | +35.36% | skill 1/1; base 1/1 |
| claude-code | build-base-default-tag | 656,983 | 246,129 | +410,854 | +166.93% | skill 1/1; base 1/1 |
| claude-code | build-base-gb300 | 658,733 | 656,783 | +1,950 | +0.30% | skill 1/1; base 1/1 |
| claude-code | build-base-gb300-release-tag | 411,896 | 563,116 | -151,220 | -26.85% | skill 1/1; base 1/1 |
| claude-code | build-base-harness-default-ref | 1,281,707 | 151,738 | +1,129,969 | +744.68% | skill 1/1; base 1/1 |
| claude-code | build-base-harness-ref | 846,147 | 183,929 | +662,218 | +360.04% | skill 1/1; base 1/1 |
| claude-code | build-base-release-tag | 990,463 | 350,357 | +640,106 | +182.70% | skill 1/1; base 1/1 |
| claude-code | build-lvs-gb300 | 619,676 | 2,387,667 | -1,767,991 | -74.05% | skill 1/1; base 1/1 |
| claude-code | build-search-gb300 | 710,129 | 212,853 | +497,276 | +233.62% | skill 1/1; base 1/1 |
| claude-code | build-search-profile | 835,005 | 2,168,199 | -1,333,194 | -61.49% | skill 1/1; base 1/1 |
| claude-code | search-running-archive | 273,319 | 186,258 | +87,061 | +46.74% | skill 1/1; base 1/1 |
| codex | All cases | 6,625,217 | 7,102,896 | -477,679 | -6.73% | skill 11/11; base 11/11 |
| codex | build-alerts-gb300 | 318,531 | 729,103 | -410,572 | -56.31% | skill 1/1; base 1/1 |
| codex | build-base-default-tag | 757,339 | 911,801 | -154,462 | -16.94% | skill 1/1; base 1/1 |
| codex | build-base-gb300 | 1,279,077 | 526,054 | +753,023 | +143.15% | skill 1/1; base 1/1 |
| codex | build-base-gb300-release-tag | 1,211,034 | 1,439,698 | -228,664 | -15.88% | skill 1/1; base 1/1 |
| codex | build-base-harness-default-ref | 1,066,438 | 245,016 | +821,422 | +335.25% | skill 1/1; base 1/1 |
| codex | build-base-harness-ref | 725,129 | 83,110 | +642,019 | +772.49% | skill 1/1; base 1/1 |
| codex | build-base-release-tag | 95,842 | 1,173,021 | -1,077,179 | -91.83% | skill 1/1; base 1/1 |
| codex | build-lvs-gb300 | 305,764 | 472,063 | -166,299 | -35.23% | skill 1/1; base 1/1 |
| codex | build-search-gb300 | 264,325 | 114,901 | +149,424 | +130.05% | skill 1/1; base 1/1 |
| codex | build-search-profile | 247,534 | 1,316,910 | -1,069,376 | -81.20% | skill 1/1; base 1/1 |
| codex | search-running-archive | 354,204 | 91,219 | +262,985 | +288.30% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 14,195,693 | 14,421,530 | -225,837 | -1.57% | skill 22/22; base 22/22 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 171 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 11 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **CRITICAL** CONTENT_DEDUP/chunk_count_limit: Tier 2 produced more than 512 content chunks. (`references/services/sop/build-ds-sop.md`)
- **MEDIUM** PII/ip_addresses: Public IP address (`references/prerequisites.md:213`)
- **MEDIUM** PII/ip_addresses: Public IP address (`references/prerequisites.md:220`)
- **MEDIUM** PII/ip_addresses: Public IP address (`references/prerequisites.md:221`)
- **MEDIUM** PII/ip_addresses: Public IP address (`references/troubleshooting.md:130`)
- 167 additional finding(s) are available in the full evaluation artifacts.

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
