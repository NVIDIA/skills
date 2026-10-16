# Skill Benchmark: tao-run-deft-aoi

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `tao-run-deft-aoi`
- Evaluation date: 2026-10-16
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 3 evaluation tasks (3 positive)
- Dataset digest: `sha256:ec1289a65b99daada0be3fb22c3130744571015a752689cf955682ed47b014f3` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 96.2% — baseline ran, but no comparable score was available; uplift unavailable | 89.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 40.0% → 100.0% (+60.0 points) | 40.0% → 86.7% (+46.7 points) |
| Discoverability | 98.3% — baseline ran, but no comparable score was available; uplift unavailable | 85.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 40.6% → 90.0% (+49.4 points) | 52.5% → 88.3% (+35.8 points) |
| Efficiency | 92.5% — baseline ran, but no comparable score was available; uplift unavailable | 89.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 602,208 | 545,786 | +56,422 | +10.34% | skill 3/3; base 3/3 |
| claude-code | tao-run-deft-aoi-basic | 71,084 | 30,373 | +40,711 | +134.04% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-aoi-customer-metric | 184,995 | 192,209 | -7,214 | -3.75% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-aoi-resume-state | 346,129 | 323,204 | +22,925 | +7.09% | skill 1/1; base 1/1 |
| codex | All cases | 380,734 | 185,932 | +194,802 | +104.77% | skill 3/3; base 3/3 |
| codex | tao-run-deft-aoi-basic | 71,005 | 13,530 | +57,475 | +424.80% | skill 1/1; base 1/1 |
| codex | tao-run-deft-aoi-customer-metric | 169,793 | 70,327 | +99,466 | +141.43% | skill 1/1; base 1/1 |
| codex | tao-run-deft-aoi-resume-state | 139,936 | 102,075 | +37,861 | +37.09% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 982,942 | 731,718 | +251,224 | +34.33% | skill 6/6; base 6/6 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 165 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 3 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/preflight.md and references/tao-mine-aoi-images.md:
  "## Pre-Flight" in references/preflight.md (lines 31-38)
  vs "## DEFT-Loop Inputs" in references/tao-mine-aoi-images.md (lines 23-32) (`references/preflight.md:31`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/applications/tao-run-deft-aoi/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/applications/tao-run-deft-aoi/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/applications/tao-run-deft-aoi/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'tags' is ignored; use 'metadata.tags' (`skills/applications/tao-run-deft-aoi/SKILL.md`)
- 161 additional finding(s) are available in the full evaluation artifacts.

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
