# Skill Benchmark: tao-train-sparse4d

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `tao-train-sparse4d`
- Evaluation date: 2026-10-16
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 7 evaluation tasks (7 positive)
- Dataset digest: `sha256:2b0115d60b2ccf67771cdf91e07158641e64d575b9497430129ea837a8b28084` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 98.5% — baseline ran, but no comparable score was available; uplift unavailable | 77.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 5.7% → 100.0% (+94.3 points) | 51.4% → 100.0% (+48.6 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 0.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 29.8% → 95.5% (+65.7 points) | 51.0% → 86.9% (+35.9 points) |
| Efficiency | 96.8% — baseline ran, but no comparable score was available; uplift unavailable | 99.4% → 99.4% (±0.0 points) |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 610,453 | 1,888,555 | -1,278,102 | -67.68% | skill 7/7; base 7/7 |
| claude-code | tao-train-sparse4d-basic | 68,595 | 30,852 | +37,743 | +122.34% | skill 1/1; base 1/1 |
| claude-code | tao-train-sparse4d-evaluate-checkpoint | 69,056 | 612,417 | -543,361 | -88.72% | skill 1/1; base 1/1 |
| claude-code | tao-train-sparse4d-missing-backbone | 66,867 | 449,067 | -382,200 | -85.11% | skill 1/1; base 1/1 |
| claude-code | tao-train-sparse4d-resnet101-ltt | 162,336 | 327,520 | -165,184 | -50.43% | skill 1/1; base 1/1 |
| claude-code | tao-train-sparse4d-resnet50 | 67,350 | 30,921 | +36,429 | +117.81% | skill 1/1; base 1/1 |
| claude-code | tao-train-sparse4d-resume | 67,219 | 405,633 | -338,414 | -83.43% | skill 1/1; base 1/1 |
| claude-code | tao-train-sparse4d-taxonomy-mismatch | 109,030 | 32,145 | +76,885 | +239.18% | skill 1/1; base 1/1 |
| codex | All cases | 99,412 | 98,635 | +777 | +0.79% | skill 7/7; base 7/7 |
| codex | tao-train-sparse4d-basic | 14,165 | 14,145 | +20 | +0.14% | skill 1/1; base 1/1 |
| codex | tao-train-sparse4d-evaluate-checkpoint | 14,185 | 14,104 | +81 | +0.57% | skill 1/1; base 1/1 |
| codex | tao-train-sparse4d-missing-backbone | 14,182 | 14,026 | +156 | +1.11% | skill 1/1; base 1/1 |
| codex | tao-train-sparse4d-resnet101-ltt | 14,305 | 14,192 | +113 | +0.80% | skill 1/1; base 1/1 |
| codex | tao-train-sparse4d-resnet50 | 14,190 | 14,202 | -12 | -0.08% | skill 1/1; base 1/1 |
| codex | tao-train-sparse4d-resume | 14,095 | 13,973 | +122 | +0.87% | skill 1/1; base 1/1 |
| codex | tao-train-sparse4d-taxonomy-mismatch | 14,290 | 13,993 | +297 | +2.12% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 709,865 | 1,987,190 | -1,277,325 | -64.28% | skill 14/14; base 14/14 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 26 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 7 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/models/tao-train-sparse4d/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/models/tao-train-sparse4d/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/models/tao-train-sparse4d/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'tags' is ignored; use 'metadata.tags' (`skills/models/tao-train-sparse4d/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/models/tao-train-sparse4d`)
- 21 additional finding(s) are available in the full evaluation artifacts.

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
