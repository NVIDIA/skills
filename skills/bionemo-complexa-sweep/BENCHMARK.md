# Skill Benchmark: complexa-sweep

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `complexa-sweep`
- Evaluation date: 2026-10-21
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 6 evaluation tasks (5 positive, 1 negative)
- Dataset digest: `sha256:b6689c4d32bb5ad0652952057f902b7baa1395cdfd9657997e9b794ba22f3510` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 89.3% — baseline ran, but no comparable score was available; uplift unavailable | 89.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 96.7% → 100.0% (+3.3 points) | 73.3% → 100.0% (+26.7 points) |
| Discoverability | 77.0% — baseline ran, but no comparable score was available; uplift unavailable | 80.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 97.6% → 97.9% (+0.3 points) | 91.7% → 93.8% (+2.1 points) |
| Efficiency | 71.7% — baseline ran, but no comparable score was available; uplift unavailable | 73.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 6,842,023 | 5,466,181 | +1,375,842 | +25.17% | skill 6/6; base 6/6 |
| claude-code | complexa-sweep-001 | 1,587,852 | 1,420,307 | +167,545 | +11.80% | skill 1/1; base 1/1 |
| claude-code | complexa-sweep-002 | 1,965,231 | 1,013,958 | +951,273 | +93.82% | skill 1/1; base 1/1 |
| claude-code | complexa-sweep-003 | 2,070,497 | 1,356,278 | +714,219 | +52.66% | skill 1/1; base 1/1 |
| claude-code | complexa-sweep-004 | 132,869 | 121,080 | +11,789 | +9.74% | skill 1/1; base 1/1 |
| claude-code | complexa-sweep-005 | 809,076 | 1,103,146 | -294,070 | -26.66% | skill 1/1; base 1/1 |
| claude-code | complexa-sweep-006 | 276,498 | 451,412 | -174,914 | -38.75% | skill 1/1; base 1/1 |
| codex | All cases | 1,636,476 | 1,095,665 | +540,811 | +49.36% | skill 6/6; base 6/6 |
| codex | complexa-sweep-001 | 422,247 | 179,167 | +243,080 | +135.67% | skill 1/1; base 1/1 |
| codex | complexa-sweep-002 | 323,265 | 194,715 | +128,550 | +66.02% | skill 1/1; base 1/1 |
| codex | complexa-sweep-003 | 493,125 | 330,649 | +162,476 | +49.14% | skill 1/1; base 1/1 |
| codex | complexa-sweep-004 | 42,563 | 95,429 | -52,866 | -55.40% | skill 1/1; base 1/1 |
| codex | complexa-sweep-005 | 183,939 | 216,005 | -32,066 | -14.85% | skill 1/1; base 1/1 |
| codex | complexa-sweep-006 | 171,337 | 79,700 | +91,637 | +114.98% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 8,478,499 | 6,561,846 | +1,916,653 | +29.21% | skill 12/12; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 26 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 6 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **CRITICAL** CONTENT_DEDUP/llm_cluster_member_limit: A Tier 2 cluster exceeds the LLM member limit. (`skills/complexa-sweep`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/complexa-sweep/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/complexa-sweep/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/complexa-sweep/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Instructions lack clear action verbs (`skills/complexa-sweep/SKILL.md`)
- 22 additional finding(s) are available in the full evaluation artifacts.

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
