# Skill Benchmark: doca-comm-channel-admin

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `doca-comm-channel-admin`
- Evaluation date: 2026-10-22
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-sonnet-5-5`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:9d2e8574f2afb5f9a860008df26c43b3364e7a22f743c6823e50820b6cd7785a` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 91.5% — baseline ran, but no comparable score was available; uplift unavailable | 90.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 20.0% → 100.0% (+80.0 points) | 55.0% → 90.0% (+35.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 26.8% → 83.8% (+57.0 points) | 24.6% → 67.9% (+43.3 points) |
| Efficiency | 73.9% — baseline ran, but no comparable score was available; uplift unavailable | 100.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 529,900 | 332,101 | N/A | N/A | skill 3/4; base 3/4 |
| claude-code | doca-comm-channel-admin.explicit-positive.v1 | 138,209 | 59,924 | +78,285 | +130.64% | skill 1/1; base 1/1 |
| claude-code | doca-comm-channel-admin.implicit-positive.v1 | 288,726 | 181,673 | +107,053 | +58.93% | skill 1/1; base 1/1 |
| claude-code | doca-comm-channel-admin.implicit-positive.v2 | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | doca-comm-channel-admin.negative.v1 | 102,965 | 90,504 | +12,461 | +13.77% | skill 1/1; base 1/1 |
| codex | All cases | 329,062 | 78,328 | N/A | N/A | skill 3/4; base 3/4 |
| codex | doca-comm-channel-admin.explicit-positive.v1 | 85,163 | 25,220 | +59,943 | +237.68% | skill 1/1; base 1/1 |
| codex | doca-comm-channel-admin.implicit-positive.v1 | 201,033 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| codex | doca-comm-channel-admin.implicit-positive.v2 | N/A | 38,965 | N/A | N/A | skill 0/1; base 1/1 |
| codex | doca-comm-channel-admin.negative.v1 | 42,866 | 14,143 | +28,723 | +203.09% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 858,962 | 410,429 | N/A | N/A | skill 6/8; base 6/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 3 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found within SKILL.md:
  "## Examples" in SKILL.md (lines 3-11)
  vs "## Example questions this skill answers well" in SKILL.md (lines 58-95) (`SKILL.md:3`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and TASKS.md:
  "## Error taxonomy" in CAPABILITIES.md (lines 101-145)
  vs "## debug" in TASKS.md (lines 174-224) (`CAPABILITIES.md:101`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across CAPABILITIES.md and SKILL.md and TASKS.md:
  "# DOCA Comm Channel Admin Tool — Capabilities" in CAPABILITIES.md (lines 1-20)
  vs "## Capabilities and modes" in CAPABILITIES.md (lines 45-87)
  vs "## Instructions" in SKILL.md (lines 12-57)
  vs "# DOCA Comm Channel Admin Tool — Tasks" in TASKS.md (lines 1-31)
  vs "## run" in TASKS.md (lines 114-147) (`CAPABILITIES.md:1`)
- **MEDIUM** QUALITY/quality_discoverability: Description contains vague words (`skills/tools/doca-comm-channel-admin/SKILL.md`)
- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'doca_version' is not kebab-case (`skills/tools/doca-comm-channel-admin/SKILL.md`)
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

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
