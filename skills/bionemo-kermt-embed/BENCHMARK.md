# Skill Benchmark: kermt-embed

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `kermt-embed`
- Evaluation date: 2026-09-15
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 5 evaluation tasks (4 positive, 1 negative)
- Dataset digest: `sha256:150b82ffdc6a29ff51ba6509f4ed70d6074e5944b957cdbd6c76f42ea048a482` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 82.0% — baseline ran, but no comparable score was available; uplift unavailable | 83.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 92.3% → 100.0% (+7.7 points) | 85.7% → 100.0% (+14.3 points) |
| Correctness | 24.6% → 84.0% (+59.4 points) | 80.0% → 84.0% (+4.0 points) |
| Discoverability | 93.8% — baseline ran, but no comparable score was available; uplift unavailable | 86.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 25.4% → 47.5% (+22.1 points) | 30.4% → 51.5% (+21.1 points) |
| Efficiency | 84.6% — baseline ran, but no comparable score was available; uplift unavailable | 94.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,189,079 | 2,993,659 | N/A | N/A | skill 5/5; base 13/13 |
| claude-code | kermt-embed-001 | 363,591 | 400,469 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | kermt-embed-002 | 285,085 | 256,226 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | kermt-embed-003 | 293,247 | 596,053 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | kermt-embed-004 | 832,919 | 248,102 | +584,817 | +235.72% | skill 1/1; base 1/1 |
| claude-code | kermt-embed-005 | 414,237 | 1,492,809 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 426,747 | 1,368,222 | N/A | N/A | skill 5/5; base 7/7 |
| codex | kermt-embed-001 | 96,323 | 299,366 | N/A | N/A | skill 1/1; base 3/3 |
| codex | kermt-embed-002 | 30,053 | 193,251 | -163,198 | -84.45% | skill 1/1; base 1/1 |
| codex | kermt-embed-003 | 86,312 | 317,836 | -231,524 | -72.84% | skill 1/1; base 1/1 |
| codex | kermt-embed-004 | 78,909 | 54,238 | +24,671 | +45.49% | skill 1/1; base 1/1 |
| codex | kermt-embed-005 | 135,150 | 503,531 | -368,381 | -73.16% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,615,826 | 4,361,881 | N/A | N/A | skill 10/10; base 20/20 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 44 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/kermt-embed/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/kermt-embed/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/kermt-embed/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/kermt-embed/SKILL.md`)
- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'risk_tier' is not kebab-case (`skills/kermt-embed/SKILL.md`)
- 39 additional finding(s) are available in the full evaluation artifacts.

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
