# Skill Benchmark: paidf-curation-and-retrieval

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `paidf-curation-and-retrieval`
- Evaluation date: 2026-09-14
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 13 evaluation tasks (13 positive)
- Dataset digest: `sha256:042681c515dffad615f864930465ed62da31beed62934ac2d384f86755134fcd` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 92.7% — baseline ran, but no comparable score was available; uplift unavailable | 87.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 90.9% → 100.0% (+9.1 points) |
| Correctness | 33.9% → 100.0% (+66.1 points) | 40.9% → 93.9% (+53.0 points) |
| Discoverability | 83.9% — baseline ran, but no comparable score was available; uplift unavailable | 68.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 36.0% → 93.2% (+57.2 points) | 34.0% → 81.4% (+47.4 points) |
| Efficiency | 86.5% — baseline ran, but no comparable score was available; uplift unavailable | 95.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,288,296 | 4,025,953 | N/A | N/A | skill 13/13; base 23/23 |
| claude-code | 1 | 194,307 | 134,270 | +60,037 | +44.71% | skill 1/1; base 1/1 |
| claude-code | 11 | 107,671 | 124,357 | -16,686 | -13.42% | skill 1/1; base 1/1 |
| claude-code | 12 | 30,519 | 30,463 | +56 | +0.18% | skill 1/1; base 1/1 |
| claude-code | 13 | 31,006 | 30,552 | +454 | +1.49% | skill 1/1; base 1/1 |
| claude-code | 14 | 152,136 | 181,484 | -29,348 | -16.17% | skill 1/1; base 1/1 |
| claude-code | 15 | 242,114 | 672,222 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | 16 | 110,461 | 551,320 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | 2 | 244,405 | 629,256 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | 3 | 676,652 | 357,964 | +318,688 | +89.03% | skill 1/1; base 1/1 |
| claude-code | 4 | 112,298 | 564,283 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | 5 | 110,976 | 230,830 | -119,854 | -51.92% | skill 1/1; base 1/1 |
| claude-code | 6 | 164,299 | 155,718 | +8,581 | +5.51% | skill 1/1; base 1/1 |
| claude-code | 9 | 111,452 | 363,234 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 1,503,166 | 2,639,826 | N/A | N/A | skill 13/13; base 22/22 |
| codex | 1 | 49,859 | 46,279 | +3,580 | +7.74% | skill 1/1; base 1/1 |
| codex | 11 | 138,660 | 227,221 | N/A | N/A | skill 1/1; base 3/3 |
| codex | 12 | 13,668 | 13,531 | +137 | +1.01% | skill 1/1; base 1/1 |
| codex | 13 | 14,294 | 14,748 | -454 | -3.08% | skill 1/1; base 1/1 |
| codex | 14 | 14,190 | 13,903 | +287 | +2.06% | skill 1/1; base 1/1 |
| codex | 15 | 90,915 | 241,015 | N/A | N/A | skill 1/1; base 3/3 |
| codex | 16 | 189,030 | 433,718 | N/A | N/A | skill 1/1; base 3/3 |
| codex | 2 | 116,304 | 453,735 | N/A | N/A | skill 1/1; base 3/3 |
| codex | 3 | 523,092 | 235,541 | +287,551 | +122.08% | skill 1/1; base 1/1 |
| codex | 4 | 95,122 | 69,636 | +25,486 | +36.60% | skill 1/1; base 1/1 |
| codex | 5 | 49,637 | 392,768 | -343,131 | -87.36% | skill 1/1; base 1/1 |
| codex | 6 | 115,754 | 101,294 | +14,460 | +14.28% | skill 1/1; base 1/1 |
| codex | 9 | 92,641 | 396,437 | N/A | N/A | skill 1/1; base 2/2 |
| ALL AGENTS | Dataset aggregate | 3,791,462 | 6,665,779 | N/A | N/A | skill 26/26; base 45/45 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 6 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 13 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in calibration-config.md (`skills/paidf-curation-and-retrieval/SKILL.md`)
- **MEDIUM** SECURITY/Unknown (SDI-2): The skill includes an 'employee_conduct_monitoring' catalog entry alongside video-curation/analytics catalogs (traffic,  (`references/context-understanding.md:106`)
- **MEDIUM** SECURITY/Unknown (SQP-2): The 'employee_conduct_monitoring' catalog is listed in a lookup table with no privacy or consent advisory, making it ind (`references/context-understanding.md:106`)
- **MEDIUM** SECURITY/Unknown (SQP-2): Phase 3 instructs the agent to drop clips from overrepresented labels and to 'move or symlink' retained clips into a new (`references/distribution-aware-curation.md:267`)
- **MEDIUM** SECURITY/Unknown (SQP-2): The skill card documents that the skill outputs shell commands and configuration instructions, and requires cloud/API cr (`skill-card.md:48`)
- 1 additional finding(s) are available in the full evaluation artifacts.

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
