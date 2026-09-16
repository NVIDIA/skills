# Skill Benchmark: kermt-add-cmim-pretrain

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `kermt-add-cmim-pretrain`
- Evaluation date: 2026-09-14
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:67190f2c3c6d387ac214662edede46001c6cf423a2cd1b21f2ba57a5584b1e42` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 87.8% — baseline ran, but no comparable score was available; uplift unavailable | 82.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 50.0% → 100.0% (+50.0 points) |
| Correctness | 13.3% → 100.0% (+86.7 points) | 91.4% → 85.0% (-6.4 points) |
| Discoverability | 99.3% — baseline ran, but no comparable score was available; uplift unavailable | 83.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 20.0% → 55.6% (+35.6 points) | 47.1% → 65.0% (+17.9 points) |
| Efficiency | 84.2% — baseline ran, but no comparable score was available; uplift unavailable | 80.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,843,834 | 1,470,837 | N/A | N/A | skill 4/4; base 9/9 |
| claude-code | kermt-add-cmim-pretrain-001 | 367,265 | 665,452 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | kermt-add-cmim-pretrain-002 | 279,378 | 344,760 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | kermt-add-cmim-pretrain-003 | 574,222 | 331,621 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | kermt-add-cmim-pretrain-004 | 622,969 | 129,004 | +493,965 | +382.91% | skill 1/1; base 1/1 |
| codex | All cases | 1,104,206 | 3,185,479 | N/A | N/A | skill 4/4; base 7/7 |
| codex | kermt-add-cmim-pretrain-001 | 123,118 | 1,434,461 | N/A | N/A | skill 1/1; base 3/3 |
| codex | kermt-add-cmim-pretrain-002 | 72,697 | 455,195 | N/A | N/A | skill 1/1; base 2/2 |
| codex | kermt-add-cmim-pretrain-003 | 655,766 | 1,189,781 | -534,015 | -44.88% | skill 1/1; base 1/1 |
| codex | kermt-add-cmim-pretrain-004 | 252,625 | 106,042 | +146,583 | +138.23% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,948,040 | 4,656,316 | N/A | N/A | skill 8/8; base 16/16 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 50 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/kermt-add-cmim-pretrain/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/kermt-add-cmim-pretrain/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/kermt-add-cmim-pretrain/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/kermt-add-cmim-pretrain/SKILL.md`)
- **MEDIUM** SCHEMA/metadata_key_style: Metadata key 'risk_tier' is not kebab-case (`skills/kermt-add-cmim-pretrain/SKILL.md`)
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
