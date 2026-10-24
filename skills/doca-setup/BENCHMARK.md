# Skill Benchmark: doca-setup

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-setup`
- Evaluation date: 2026-10-24
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:169f7c0489e994502eee43ed5716bf51cebe677b1bd8f3a68ecc6278038d804e` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 90.9% — baseline ran, but no comparable score was available; uplift unavailable | 86.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 60.0% → 95.0% (+35.0 points) | 70.0% → 85.0% (+15.0 points) |
| Discoverability | 96.7% — baseline ran, but no comparable score was available; uplift unavailable | 88.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 50.2% → 74.4% (+24.2 points) | 66.3% → 81.3% (+15.0 points) |
| Efficiency | 88.2% — baseline ran, but no comparable score was available; uplift unavailable | 77.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,049,186 | 430,489 | +618,697 | +143.72% | skill 4/4; base 4/4 |
| claude-code | doca-setup.explicit-positive.v1 | 334,793 | 93,213 | +241,580 | +259.17% | skill 1/1; base 1/1 |
| claude-code | doca-setup.implicit-positive.v1 | 329,261 | 124,012 | +205,249 | +165.51% | skill 1/1; base 1/1 |
| claude-code | doca-setup.implicit-positive.v2 | 226,275 | 92,481 | +133,794 | +144.67% | skill 1/1; base 1/1 |
| claude-code | doca-setup.negative.v1 | 158,857 | 120,783 | +38,074 | +31.52% | skill 1/1; base 1/1 |
| codex | All cases | 275,897 | 147,229 | +128,668 | +87.39% | skill 4/4; base 4/4 |
| codex | doca-setup.explicit-positive.v1 | 128,398 | 67,422 | +60,976 | +90.44% | skill 1/1; base 1/1 |
| codex | doca-setup.implicit-positive.v1 | 80,630 | 26,891 | +53,739 | +199.84% | skill 1/1; base 1/1 |
| codex | doca-setup.implicit-positive.v2 | 53,254 | 39,572 | +13,682 | +34.57% | skill 1/1; base 1/1 |
| codex | doca-setup.negative.v1 | 13,615 | 13,344 | +271 | +2.03% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,325,083 | 577,718 | +747,365 | +129.37% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 14 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 3 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across TASKS.md and references/command-appendix.md:
  "## Deferred task verbs" in TASKS.md (lines 263-268)
  vs "## Deferred task verbs" in references/command-appendix.md (lines 33-38) (`TASKS.md:263`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and TASKS.md and references/recognize.md:
  "# DOCA setup" in SKILL.md (lines 1-40)
  vs "## Instructions" in SKILL.md (lines 49-87)
  vs "## When to load this skill" in SKILL.md (lines 118-133)
  vs "## What this skill provides" in SKILL.md (lines 134-142)
  vs "## Related skills" in SKILL.md (lines 186-192)
  vs "# DOCA setup workflows" in TASKS.md (lines 1-24)
  vs "## recognize" in references/recognize.md (lines 2-2) (`SKILL.md:1`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found within CAPABILITIES.md:
  "## Error taxonomy" in CAPABILITIES.md (lines 83-83)
  vs "## Error taxonomy" in CAPABILITIES.md (lines 96-96) (`CAPABILITIES.md:83`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in command-appendix.md (`skills/doca-setup/SKILL.md`)
- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'doca_version' is not kebab-case (`skills/doca-setup/SKILL.md`)
- 12 additional finding(s) are available in the full evaluation artifacts.

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
