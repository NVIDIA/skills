# Skill Benchmark: ethos-orchestrate

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `ethos-orchestrate`
- Evaluation date: 2026-10-16
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:a405e089d93c5fbaef1a4083da0609f55595a62b10d5fc3c59d9ce536e49b4d1` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 95.7% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 75.0% → 100.0% (+25.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 70.0% → 100.0% (+30.0 points) | 70.0% → 95.0% (+25.0 points) |
| Discoverability | 86.7% — baseline ran, but no comparable score was available; uplift unavailable | 91.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 61.0% → 96.3% (+35.3 points) | 51.7% → 96.7% (+45.0 points) |
| Efficiency | 95.5% — baseline ran, but no comparable score was available; uplift unavailable | 91.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,302,367 | 2,806,171 | -1,503,804 | -53.59% | skill 4/4; base 4/4 |
| claude-code | orchestrate-explicit-new-ethos | 734,212 | 585,182 | +149,030 | +25.47% | skill 1/1; base 1/1 |
| claude-code | orchestrate-implicit-existing-status | 233,175 | 251,638 | -18,463 | -7.34% | skill 1/1; base 1/1 |
| claude-code | orchestrate-legacy-rename | 305,800 | 1,940,368 | -1,634,568 | -84.24% | skill 1/1; base 1/1 |
| claude-code | orchestrate-negative | 29,180 | 28,983 | +197 | +0.68% | skill 1/1; base 1/1 |
| codex | All cases | 346,265 | 455,748 | -109,483 | -24.02% | skill 4/4; base 4/4 |
| codex | orchestrate-explicit-new-ethos | 29,475 | 152,903 | -123,428 | -80.72% | skill 1/1; base 1/1 |
| codex | orchestrate-implicit-existing-status | 100,580 | 124,371 | -23,791 | -19.13% | skill 1/1; base 1/1 |
| codex | orchestrate-legacy-rename | 202,838 | 165,257 | +37,581 | +22.74% | skill 1/1; base 1/1 |
| codex | orchestrate-negative | 13,372 | 13,217 | +155 | +1.17% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,648,632 | 3,261,919 | -1,613,287 | -49.46% | skill 8/8; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 7 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`team-skills/ethos/ethos-orchestrate/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`team-skills/ethos/ethos-orchestrate/SKILL.md`)
- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`team-skills/ethos/ethos-orchestrate/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`team-skills/ethos/ethos-orchestrate/SKILL.md`)
- **MEDIUM** SCHEMA/author_missing: Author not specified in metadata (`team-skills/ethos/ethos-orchestrate/SKILL.md`)
- 2 additional finding(s) are available in the full evaluation artifacts.

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
