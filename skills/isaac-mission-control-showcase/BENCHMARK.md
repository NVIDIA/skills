# Skill Benchmark: isaac-mission-control-showcase

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `isaac-mission-control-showcase`
- Evaluation date: 2026-09-17
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:43742a466533f3d0d814ea60502c8cf27d813445291cbbfda6f2f931faed3fe2` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 93.1% — baseline ran, but no comparable score was available; uplift unavailable | 65.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 41.7% (-58.3 points) |
| Correctness | 25.0% → 100.0% (+75.0 points) | 22.2% → 70.0% (+47.8 points) |
| Discoverability | 90.0% — baseline ran, but no comparable score was available; uplift unavailable | 82.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 31.6% → 88.1% (+56.5 points) | 25.6% → 49.2% (+23.6 points) |
| Efficiency | 87.2% — baseline ran, but no comparable score was available; uplift unavailable | 85.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,532,910 | 1,617,779 | N/A | N/A | skill 4/4; base 8/8 |
| claude-code | isaac-mission-control-showcase-reasoning-isaac-sim-not-installed | 666,432 | 487,827 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | isaac-mission-control-showcase-reasoning-swap-resize-refusal | 199,093 | 319,336 | -120,243 | -37.65% | skill 1/1; base 1/1 |
| claude-code | isaac-mission-control-showcase-trigger-kit-extension-negative | 61,433 | 30,563 | +30,870 | +101.00% | skill 1/1; base 1/1 |
| claude-code | isaac-mission-control-showcase-trigger-run-demo | 605,952 | 780,053 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 1,358,777 | 1,029,948 | N/A | N/A | skill 6/6; base 9/9 |
| codex | isaac-mission-control-showcase-reasoning-isaac-sim-not-installed | 355,064 | 392,450 | N/A | N/A | skill 1/1; base 3/3 |
| codex | isaac-mission-control-showcase-reasoning-swap-resize-refusal | 568,072 | 204,911 | N/A | N/A | skill 3/3; base 2/2 |
| codex | isaac-mission-control-showcase-trigger-kit-extension-negative | 60,734 | 50,550 | +10,184 | +20.15% | skill 1/1; base 1/1 |
| codex | isaac-mission-control-showcase-trigger-run-demo | 374,907 | 382,037 | N/A | N/A | skill 1/1; base 3/3 |
| ALL AGENTS | Dataset aggregate | 2,891,687 | 2,647,727 | N/A | N/A | skill 10/10; base 17/17 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 49 finding(s) |
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
- 45 additional finding(s) are available in the full evaluation artifacts.

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
