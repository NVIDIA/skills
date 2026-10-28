# Skill Benchmark: pva-doctor

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `pva-doctor`
- Evaluation date: 2026-10-22
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-sonnet-5-5`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 5 evaluation tasks (4 positive, 1 negative)
- Dataset digest: `sha256:1e939133a9eb08dc70233be230625ddad334942ffd8ab02383d50cecd32ba340` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 95.0% — baseline ran, but no comparable score was available; uplift unavailable | 92.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 36.0% → 100.0% (+64.0 points) | 48.0% → 96.0% (+48.0 points) |
| Discoverability | 96.3% — baseline ran, but no comparable score was available; uplift unavailable | 91.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 45.9% → 97.1% (+51.2 points) | 61.1% → 97.1% (+36.0 points) |
| Efficiency | 81.7% — baseline ran, but no comparable score was available; uplift unavailable | 75.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,052,643 | 867,925 | +184,718 | +21.28% | skill 5/5; base 5/5 |
| claude-code | doctor-missing-sdk | 243,285 | 254,585 | -11,300 | -4.44% | skill 1/1; base 1/1 |
| claude-code | doctor-missing-solutions-source | 235,840 | 198,035 | +37,805 | +19.09% | skill 1/1; base 1/1 |
| claude-code | doctor-negative | 29,564 | 29,399 | +165 | +0.56% | skill 1/1; base 1/1 |
| claude-code | doctor-overlapping-output | 246,972 | 157,390 | +89,582 | +56.92% | skill 1/1; base 1/1 |
| claude-code | doctor-scoped-prerequisites | 296,982 | 228,516 | +68,466 | +29.96% | skill 1/1; base 1/1 |
| codex | All cases | 445,176 | 616,764 | -171,588 | -27.82% | skill 5/5; base 5/5 |
| codex | doctor-missing-sdk | 114,872 | 187,259 | -72,387 | -38.66% | skill 1/1; base 1/1 |
| codex | doctor-missing-solutions-source | 99,246 | 79,371 | +19,875 | +25.04% | skill 1/1; base 1/1 |
| codex | doctor-negative | 13,511 | 13,401 | +110 | +0.82% | skill 1/1; base 1/1 |
| codex | doctor-overlapping-output | 89,697 | 119,614 | -29,917 | -25.01% | skill 1/1; base 1/1 |
| codex | doctor-scoped-prerequisites | 127,850 | 217,119 | -89,269 | -41.12% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,497,819 | 1,484,689 | +13,130 | +0.88% | skill 10/10; base 10/10 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 20 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** DUPLICATE/duplicate: Duplicate content found across SKILL.md and assets/doctor-report-template.md and references/reference-maintenance.md:
  "(preamble)" in SKILL.md (lines 1-4)
  vs "(preamble)" in assets/doctor-report-template.md (lines 1-4)
  vs "(preamble)" in references/reference-maintenance.md (lines 1-4) (`SKILL.md:1`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/pva-doctor/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/pva-doctor/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/pva-doctor/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/pva-doctor/SKILL.md`)
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
