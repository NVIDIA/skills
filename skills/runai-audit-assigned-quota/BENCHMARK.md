# Skill Benchmark: runai-audit-assigned-quota

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `runai-audit-assigned-quota`
- Validation status: `passed`
- Evaluation date: 2026-10-28
- Evaluator version: `1.6.2`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 12 evaluation tasks (5 positive, 7 negative)
- Dataset digest: `sha256:833313ade3da9598cf390d2adb4483512b9dca520199d8c1c6bf268e180b75e3` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| runai-audit-assigned-quota | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| runai-audit-assigned-quota | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| runai-audit-assigned-quota | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| runai-audit-assigned-quota | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| runai-audit-assigned-quota | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |

- Controls for `runai-audit-assigned-quota`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 90.0% — baseline ran, but no comparable score was available; uplift unavailable | 83.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 86.7% → 86.7% (±0.0 points) | 83.3% → 81.7% (-1.6 points) |
| Discoverability | 88.0% — baseline ran, but no comparable score was available; uplift unavailable | 90.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 80.1% → 84.7% (+4.6 points) | 87.5% → 84.7% (-2.8 points) |
| Efficiency | 90.6% — baseline ran, but no comparable score was available; uplift unavailable | 59.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 96.0% → 94.5% (-1.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 92.2% → 91.3% (-0.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,038,556 | 1,517,343 | +521,213 | +34.35% | skill 12/12; base 12/12 |
| claude-code | boundary-no-k8s-fallback | 147,978 | 147,049 | +929 | +0.63% | skill 1/1; base 1/1 |
| claude-code | neg-cluster-health | 159,042 | 152,377 | +6,665 | +4.37% | skill 1/1; base 1/1 |
| claude-code | neg-count-running | 72,417 | 71,904 | +513 | +0.71% | skill 1/1; base 1/1 |
| claude-code | neg-failed-workload | 149,715 | 146,914 | +2,801 | +1.91% | skill 1/1; base 1/1 |
| claude-code | neg-idle-held | 113,836 | 154,193 | -40,357 | -26.17% | skill 1/1; base 1/1 |
| claude-code | neg-rebalance-write | 74,971 | 112,885 | -37,914 | -33.59% | skill 1/1; base 1/1 |
| claude-code | neg-unrelated | 111,803 | 109,795 | +2,008 | +1.83% | skill 1/1; base 1/1 |
| claude-code | neg-waste-cycles | 157,398 | 155,888 | +1,510 | +0.97% | skill 1/1; base 1/1 |
| claude-code | over-quota-pool | 260,561 | 72,583 | +187,978 | +258.98% | skill 1/1; base 1/1 |
| claude-code | quota-audit-fairness | 263,994 | 119,667 | +144,327 | +120.61% | skill 1/1; base 1/1 |
| claude-code | shared-fairly | 263,270 | 157,883 | +105,387 | +66.75% | skill 1/1; base 1/1 |
| claude-code | who-starved-where-reclaim | 263,571 | 116,205 | +147,366 | +126.82% | skill 1/1; base 1/1 |
| codex | All cases | 2,233,065 | 1,422,646 | +810,419 | +56.97% | skill 12/12; base 12/12 |
| codex | boundary-no-k8s-fallback | 49,443 | 57,077 | -7,634 | -13.37% | skill 1/1; base 1/1 |
| codex | neg-cluster-health | 190,121 | 226,834 | -36,713 | -16.18% | skill 1/1; base 1/1 |
| codex | neg-count-running | 28,051 | 27,699 | +352 | +1.27% | skill 1/1; base 1/1 |
| codex | neg-failed-workload | 29,667 | 43,251 | -13,584 | -31.41% | skill 1/1; base 1/1 |
| codex | neg-idle-held | 57,084 | 44,581 | +12,503 | +28.05% | skill 1/1; base 1/1 |
| codex | neg-rebalance-write | 106,989 | 56,415 | +50,574 | +89.65% | skill 1/1; base 1/1 |
| codex | neg-unrelated | 57,878 | 57,088 | +790 | +1.38% | skill 1/1; base 1/1 |
| codex | neg-waste-cycles | 138,701 | 211,436 | -72,735 | -34.40% | skill 1/1; base 1/1 |
| codex | over-quota-pool | 371,722 | 27,977 | +343,745 | +1228.67% | skill 1/1; base 1/1 |
| codex | quota-audit-fairness | 401,445 | 227,856 | +173,589 | +76.18% | skill 1/1; base 1/1 |
| codex | shared-fairly | 401,121 | 231,033 | +170,088 | +73.62% | skill 1/1; base 1/1 |
| codex | who-starved-where-reclaim | 400,843 | 211,399 | +189,444 | +89.61% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 4,271,621 | 2,939,989 | +1,331,632 | +45.29% | skill 24/24; base 24/24 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 8 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 12 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/quota-accounting.md:
  "### 2. Detect *genuine* pending demand (for STARVED)" in SKILL.md (lines 58-76)
  vs "## STARVED needs the pending *reason*, not just a pending count" in references/quota-accounting.md (lines 51-65) (`SKILL.md:58`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/quota-accounting.md:
  "### 6. Diagnose each STARVED project — cause drives the fix" in SKILL.md (lines 96-113)
  vs "### 7. Compose" in SKILL.md (lines 114-123)
  vs "## Examples — what a good answer looks like" in SKILL.md (lines 124-139)
  vs "## Fairness, borrowing, reclaim" in references/quota-accounting.md (lines 24-38)
  vs "## Two causes of starvation — they need different fixes" in references/quota-accounting.md (lines 66-88) (`SKILL.md:96`)
- **LOW** QUALITY/quality_discoverability: Description very long (651 chars, recommend 50-150) (`skills/runai-audit-assigned-quota/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Broad description without negative triggers may cause over-triggering (`skills/runai-audit-assigned-quota/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/runai-audit-assigned-quota/SKILL.md`)
- 5 additional finding(s) are available in the full evaluation artifacts.

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
