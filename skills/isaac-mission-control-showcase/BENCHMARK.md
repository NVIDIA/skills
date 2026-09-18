# Skill Benchmark: isaac-mission-control-showcase

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `isaac-mission-control-showcase`
- Evaluation date: 2026-09-18
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:53df5054895d6c1cfa04e19d9145af4857e6310543c98498d2c49226247b9259` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 95.4% — baseline ran, but no comparable score was available; uplift unavailable | 83.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 43.8% → 87.5% (+43.7 points) |
| Correctness | 37.5% → 100.0% (+62.5 points) | 40.0% → 80.0% (+40.0 points) |
| Discoverability | 99.3% — baseline ran, but no comparable score was available; uplift unavailable | 86.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 32.5% → 91.9% (+59.4 points) | 40.6% → 76.9% (+36.3 points) |
| Efficiency | 86.0% — baseline ran, but no comparable score was available; uplift unavailable | 87.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,449,865 | 1,750,936 | N/A | N/A | skill 4/4; base 8/8 |
| claude-code | isaac-mission-control-showcase-reasoning-isaac-sim-not-installed | 618,559 | 714,395 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | isaac-mission-control-showcase-reasoning-swap-resize-refusal | 155,688 | 225,259 | -69,571 | -30.88% | skill 1/1; base 1/1 |
| claude-code | isaac-mission-control-showcase-trigger-kit-extension-negative | 91,755 | 30,727 | +61,028 | +198.61% | skill 1/1; base 1/1 |
| claude-code | isaac-mission-control-showcase-trigger-run-demo | 583,863 | 780,555 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 554,529 | 1,697,381 | N/A | N/A | skill 4/4; base 8/8 |
| codex | isaac-mission-control-showcase-reasoning-isaac-sim-not-installed | 236,022 | 992,441 | N/A | N/A | skill 1/1; base 3/3 |
| codex | isaac-mission-control-showcase-reasoning-swap-resize-refusal | 191,836 | 409,673 | N/A | N/A | skill 1/1; base 3/3 |
| codex | isaac-mission-control-showcase-trigger-kit-extension-negative | 42,112 | 41,084 | +1,028 | +2.50% | skill 1/1; base 1/1 |
| codex | isaac-mission-control-showcase-trigger-run-demo | 84,559 | 254,183 | -169,624 | -66.73% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,004,394 | 3,448,317 | N/A | N/A | skill 8/8; base 16/16 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 71 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/bring-up-cloud-stack/README.md and references/change-fleet-composition/README.md and references/change-map/README.md and references/isaac-sim-remote/README.md:
  "## Blockers and return behavior" in references/bring-up-cloud-stack/README.md (lines 49-56)
  vs "## Blockers and return behavior" in references/change-fleet-composition/README.md (lines 48-55)
  vs "## Blockers and return behavior" in references/change-map/README.md (lines 48-55)
  vs "## Blockers and return behavior" in references/isaac-sim-remote/README.md (lines 49-56) (`references/bring-up-cloud-stack/README.md:49`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/isaac-mission-control-showcase/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/isaac-mission-control-showcase/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/isaac-mission-control-showcase/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/isaac-mission-control-showcase/SKILL.md`)
- 67 additional finding(s) are available in the full evaluation artifacts.

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
