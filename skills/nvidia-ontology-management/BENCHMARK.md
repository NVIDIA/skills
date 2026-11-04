# Skill Benchmark: nvidia-ontology-management

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvidia-ontology-management`
- Evaluation date: 2026-10-08
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 13 evaluation tasks (13 positive)
- Dataset digest: `sha256:4c8455d88fc45fd16ded90344c740ea7149fe29edbca7f112b346c21177d23fa` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 94.9% — baseline ran, but no comparable score was available; uplift unavailable | 85.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 84.6% → 76.9% (-7.7 points) |
| Correctness | 29.2% → 100.0% (+70.8 points) | 47.7% → 89.2% (+41.5 points) |
| Discoverability | 93.7% — baseline ran, but no comparable score was available; uplift unavailable | 85.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 42.4% → 89.3% (+46.9 points) | 33.9% → 81.1% (+47.2 points) |
| Efficiency | 91.4% — baseline ran, but no comparable score was available; uplift unavailable | 93.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,136,288 | 2,211,810 | -75,522 | -3.41% | skill 13/13; base 13/13 |
| claude-code | management-eval-001-mcp-readonly | 176,205 | 177,943 | -1,738 | -0.98% | skill 1/1; base 1/1 |
| claude-code | management-eval-002-validate-422 | 91,346 | 120,663 | -29,317 | -24.30% | skill 1/1; base 1/1 |
| claude-code | management-eval-003-reset-is-destructive | 97,849 | 150,421 | -52,572 | -34.95% | skill 1/1; base 1/1 |
| claude-code | management-eval-004-token-acts-as-owner | 95,573 | 119,914 | -24,341 | -20.30% | skill 1/1; base 1/1 |
| claude-code | management-eval-005-import-replace | 99,977 | 121,086 | -21,109 | -17.43% | skill 1/1; base 1/1 |
| claude-code | management-eval-006-lineage-semantic-layer | 285,753 | 119,687 | +166,066 | +138.75% | skill 1/1; base 1/1 |
| claude-code | management-eval-007-patch-vs-put-sql-attribute | 95,844 | 220,722 | -124,878 | -56.58% | skill 1/1; base 1/1 |
| claude-code | management-eval-008-run-grain-measure | 95,583 | 30,926 | +64,657 | +209.07% | skill 1/1; base 1/1 |
| claude-code | management-eval-009-relationship-evidence | 247,442 | 247,503 | -61 | -0.02% | skill 1/1; base 1/1 |
| claude-code | management-eval-010-import-exact-readback | 97,134 | 216,736 | -119,602 | -55.18% | skill 1/1; base 1/1 |
| claude-code | management-eval-011-result-publication-mode | 161,713 | 317,242 | -155,529 | -49.03% | skill 1/1; base 1/1 |
| claude-code | management-eval-012-promotion-readback-failure | 133,068 | 183,802 | -50,734 | -27.60% | skill 1/1; base 1/1 |
| claude-code | management-eval-013-composed-lifecycle | 458,801 | 185,165 | +273,636 | +147.78% | skill 1/1; base 1/1 |
| codex | All cases | 2,393,113 | 2,250,240 | +142,873 | +6.35% | skill 13/13; base 13/13 |
| codex | management-eval-001-mcp-readonly | 797,280 | 452,293 | +344,987 | +76.28% | skill 1/1; base 1/1 |
| codex | management-eval-002-validate-422 | 57,538 | 54,886 | +2,652 | +4.83% | skill 1/1; base 1/1 |
| codex | management-eval-003-reset-is-destructive | 57,904 | 54,809 | +3,095 | +5.65% | skill 1/1; base 1/1 |
| codex | management-eval-004-token-acts-as-owner | 45,901 | 54,971 | -9,070 | -16.50% | skill 1/1; base 1/1 |
| codex | management-eval-005-import-replace | 161,338 | 55,064 | +106,274 | +193.00% | skill 1/1; base 1/1 |
| codex | management-eval-006-lineage-semantic-layer | 85,423 | 98,141 | -12,718 | -12.96% | skill 1/1; base 1/1 |
| codex | management-eval-007-patch-vs-put-sql-attribute | 45,034 | 65,171 | -20,137 | -30.90% | skill 1/1; base 1/1 |
| codex | management-eval-008-run-grain-measure | 44,493 | 54,668 | -10,175 | -18.61% | skill 1/1; base 1/1 |
| codex | management-eval-009-relationship-evidence | 413,138 | 83,561 | +329,577 | +394.41% | skill 1/1; base 1/1 |
| codex | management-eval-010-import-exact-readback | 44,648 | 55,235 | -10,587 | -19.17% | skill 1/1; base 1/1 |
| codex | management-eval-011-result-publication-mode | 118,995 | 475,882 | -356,887 | -74.99% | skill 1/1; base 1/1 |
| codex | management-eval-012-promotion-readback-failure | 110,830 | 262,196 | -151,366 | -57.73% | skill 1/1; base 1/1 |
| codex | management-eval-013-composed-lifecycle | 410,591 | 483,363 | -72,772 | -15.06% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 4,529,401 | 4,462,050 | +67,351 | +1.51% | skill 26/26; base 26/26 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 5 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 13 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: Resource-based skill lacks documentation of available resources (`skills/nvidia-ontology-management/SKILL.md`)
- **MEDIUM** QUALITY/quality_reliability: MCP skill lacks connection/error guidance (`skills/nvidia-ontology-management/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in modeling.md (`skills/nvidia-ontology-management/SKILL.md`)
- **LOW** QUALITY/quality_reliability: Inputs are used but no dedicated Inputs section is documented (`skills/nvidia-ontology-management/SKILL.md`)
- **LOW** QUALITY/quality_reliability: Structured output is used but no dedicated Output Format section is documented (`skills/nvidia-ontology-management/SKILL.md`)

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
