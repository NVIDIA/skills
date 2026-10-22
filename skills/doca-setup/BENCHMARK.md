# Skill Benchmark: doca-setup

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-setup`
- Evaluation date: 2026-10-22
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-sonnet-5-5`), Codex (`openai/openai/gpt-5.5`)
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
| Overall | 93.9% — baseline ran, but no comparable score was available; uplift unavailable | 92.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 65.0% → 100.0% (+35.0 points) | 70.0% → 95.0% (+25.0 points) |
| Discoverability | 98.3% — baseline ran, but no comparable score was available; uplift unavailable | 90.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 54.6% → 85.0% (+30.4 points) | 54.2% → 88.8% (+34.6 points) |
| Efficiency | 86.3% — baseline ran, but no comparable score was available; uplift unavailable | 86.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 786,057 | 244,936 | +541,121 | +220.92% | skill 4/4; base 4/4 |
| claude-code | doca-setup.explicit-positive.v1 | 247,952 | 60,900 | +187,052 | +307.15% | skill 1/1; base 1/1 |
| claude-code | doca-setup.implicit-positive.v1 | 222,754 | 32,673 | +190,081 | +581.77% | skill 1/1; base 1/1 |
| claude-code | doca-setup.implicit-positive.v2 | 223,260 | 61,455 | +161,805 | +263.29% | skill 1/1; base 1/1 |
| claude-code | doca-setup.negative.v1 | 92,091 | 89,908 | +2,183 | +2.43% | skill 1/1; base 1/1 |
| codex | All cases | 377,554 | 156,851 | +220,703 | +140.71% | skill 4/4; base 4/4 |
| codex | doca-setup.explicit-positive.v1 | 89,631 | 32,246 | +57,385 | +177.96% | skill 1/1; base 1/1 |
| codex | doca-setup.implicit-positive.v1 | 72,923 | 26,191 | +46,732 | +178.43% | skill 1/1; base 1/1 |
| codex | doca-setup.implicit-positive.v2 | 101,477 | 29,232 | +72,245 | +247.14% | skill 1/1; base 1/1 |
| codex | doca-setup.negative.v1 | 113,523 | 69,182 | +44,341 | +64.09% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,163,611 | 401,787 | +761,824 | +189.61% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 14 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

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
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in hardware-binding.md (`skills/doca-setup/SKILL.md`)
- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'doca_version' is not kebab-case (`skills/doca-setup/SKILL.md`)
- **MEDIUM** SECURITY/Unknown (RP1): MCP Rug Pull: Docker image references without a specific tag (:latest is implicit) or digest (@sha256:...) can be silently replaced by (`TASKS.md:185`)
- 11 additional finding(s) are available in the full evaluation artifacts.

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
