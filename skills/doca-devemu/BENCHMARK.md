# Skill Benchmark: doca-devemu

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-devemu`
- Evaluation date: 2026-10-24
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:bddc3102319aeedfab57309e910e03e8c1185f748c84845eadf3d5b4a3799de2` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 94.4% — baseline ran, but no comparable score was available; uplift unavailable | 89.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 60.0% → 100.0% (+40.0 points) | 80.0% → 95.0% (+15.0 points) |
| Discoverability | 96.7% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 55.0% → 85.6% (+30.6 points) | 75.0% → 88.8% (+13.8 points) |
| Efficiency | 89.9% — baseline ran, but no comparable score was available; uplift unavailable | 67.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 929,509 | 408,518 | +520,991 | +127.53% | skill 4/4; base 4/4 |
| claude-code | doca-devemu.explicit-positive.v1 | 239,941 | 95,729 | +144,212 | +150.65% | skill 1/1; base 1/1 |
| claude-code | doca-devemu.implicit-positive.v1 | 313,276 | 97,204 | +216,072 | +222.29% | skill 1/1; base 1/1 |
| claude-code | doca-devemu.implicit-positive.v2 | 313,194 | 125,259 | +187,935 | +150.04% | skill 1/1; base 1/1 |
| claude-code | doca-devemu.negative.v1 | 63,098 | 90,326 | -27,228 | -30.14% | skill 1/1; base 1/1 |
| codex | All cases | 377,349 | 196,511 | +180,838 | +92.02% | skill 4/4; base 4/4 |
| codex | doca-devemu.explicit-positive.v1 | 107,477 | 67,379 | +40,098 | +59.51% | skill 1/1; base 1/1 |
| codex | doca-devemu.implicit-positive.v1 | 111,376 | 48,172 | +63,204 | +131.20% | skill 1/1; base 1/1 |
| codex | doca-devemu.implicit-positive.v2 | 119,524 | 31,439 | +88,085 | +280.18% | skill 1/1; base 1/1 |
| codex | doca-devemu.negative.v1 | 38,972 | 49,521 | -10,549 | -21.30% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,306,858 | 605,029 | +701,829 | +116.00% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 17 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 4 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **CRITICAL** CONTENT_DEDUP/llm_error: LLM analysis failed for a content cluster (`skills/libs/doca-devemu`)
- **CRITICAL** CONTENT_DEDUP/llm_error: LLM analysis failed for a content cluster (`skills/libs/doca-devemu`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and TASKS.md:
  "## Observability" in CAPABILITIES.md (lines 295-302)
  vs "## DOCA 3.5 field notes" in TASKS.md (lines 83-86) (`CAPABILITIES.md:295`)
- **MEDIUM** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and references/details.md:
  "## Pattern overview" in CAPABILITIES.md (lines 40-59)
  vs "## Safety policy" in CAPABILITIES.md (lines 353-363)
  vs "## Related skills" in references/details.md (lines 177-184) (`CAPABILITIES.md:40`)
- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/libs/doca-devemu/SKILL.md`)
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
