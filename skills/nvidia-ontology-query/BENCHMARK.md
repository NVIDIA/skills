# Skill Benchmark: nvidia-ontology-query

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvidia-ontology-query`
- Evaluation date: 2026-10-08
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 13 evaluation tasks (13 positive)
- Dataset digest: `sha256:314823168b2eada85a48a450189c9d2cc1d374eed343c106147bd2b63080bf72` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 95.3% — baseline ran, but no comparable score was available; uplift unavailable | 80.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 84.6% → 92.3% (+7.7 points) |
| Correctness | 47.7% → 100.0% (+52.3 points) | 50.8% → 70.8% (+20.0 points) |
| Discoverability | 84.6% — baseline ran, but no comparable score was available; uplift unavailable | 75.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 55.5% → 96.0% (+40.5 points) | 48.8% → 68.5% (+19.7 points) |
| Efficiency | 95.6% — baseline ran, but no comparable score was available; uplift unavailable | 95.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,646,283 | 5,779,380 | -4,133,097 | -71.51% | skill 13/13; base 13/13 |
| claude-code | query-eval-001-mcp-first | 96,646 | 218,379 | -121,733 | -55.74% | skill 1/1; base 1/1 |
| claude-code | query-eval-002-no-fastapi-public | 97,882 | 29,472 | +68,410 | +232.12% | skill 1/1; base 1/1 |
| claude-code | query-eval-003-conversation-409 | 61,587 | 91,478 | -29,891 | -32.68% | skill 1/1; base 1/1 |
| claude-code | query-eval-004-sso-vs-api-token | 95,201 | 122,413 | -27,212 | -22.23% | skill 1/1; base 1/1 |
| claude-code | query-eval-005-no-raw-schema-tools | 61,522 | 29,540 | +31,982 | +108.27% | skill 1/1; base 1/1 |
| claude-code | query-eval-006-cuopt-not-in-repo | 624,192 | 4,985,881 | -4,361,689 | -87.48% | skill 1/1; base 1/1 |
| claude-code | query-eval-007-readiness-before-retry | 165,649 | 29,437 | +136,212 | +462.72% | skill 1/1; base 1/1 |
| claude-code | query-eval-008-quantity-not-ratio | 30,293 | 30,144 | +149 | +0.49% | skill 1/1; base 1/1 |
| claude-code | query-eval-009-run-metric-fanout | 96,128 | 30,591 | +65,537 | +214.24% | skill 1/1; base 1/1 |
| claude-code | query-eval-010-scenario-lineage | 95,933 | 122,173 | -26,240 | -21.48% | skill 1/1; base 1/1 |
| claude-code | query-eval-011-prose-versus-rows | 95,264 | 29,639 | +65,625 | +221.41% | skill 1/1; base 1/1 |
| claude-code | query-eval-012-source-role | 30,645 | 30,615 | +30 | +0.10% | skill 1/1; base 1/1 |
| claude-code | query-eval-013-truncated-result | 95,341 | 29,618 | +65,723 | +221.90% | skill 1/1; base 1/1 |
| codex | All cases | 1,088,744 | 5,171,383 | -4,082,639 | -78.95% | skill 13/13; base 13/13 |
| codex | query-eval-001-mcp-first | 44,904 | 62,766 | -17,862 | -28.46% | skill 1/1; base 1/1 |
| codex | query-eval-002-no-fastapi-public | 13,385 | 69,411 | -56,026 | -80.72% | skill 1/1; base 1/1 |
| codex | query-eval-003-conversation-409 | 126,942 | 263,292 | -136,350 | -51.79% | skill 1/1; base 1/1 |
| codex | query-eval-004-sso-vs-api-token | 44,764 | 59,779 | -15,015 | -25.12% | skill 1/1; base 1/1 |
| codex | query-eval-005-no-raw-schema-tools | 28,732 | 45,391 | -16,659 | -36.70% | skill 1/1; base 1/1 |
| codex | query-eval-006-cuopt-not-in-repo | 491,109 | 4,405,244 | -3,914,135 | -88.85% | skill 1/1; base 1/1 |
| codex | query-eval-007-readiness-before-retry | 28,709 | 13,336 | +15,373 | +115.27% | skill 1/1; base 1/1 |
| codex | query-eval-008-quantity-not-ratio | 13,516 | 13,452 | +64 | +0.48% | skill 1/1; base 1/1 |
| codex | query-eval-009-run-metric-fanout | 44,971 | 13,686 | +31,285 | +228.59% | skill 1/1; base 1/1 |
| codex | query-eval-010-scenario-lineage | 45,697 | 83,805 | -38,108 | -45.47% | skill 1/1; base 1/1 |
| codex | query-eval-011-prose-versus-rows | 44,850 | 13,478 | +31,372 | +232.76% | skill 1/1; base 1/1 |
| codex | query-eval-012-source-role | 132,302 | 114,167 | +18,135 | +15.88% | skill 1/1; base 1/1 |
| codex | query-eval-013-truncated-result | 28,863 | 13,576 | +15,287 | +112.60% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,735,027 | 10,950,763 | -8,215,736 | -75.02% | skill 26/26; base 26/26 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 2 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 13 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SECURITY/Unknown (SQP-2): The documentation explicitly states that the `aud` (audience) claim is NOT checked on SSO bearer tokens forwarded by AI- (`references/stack.md:14`)
- **LOW** SECURITY/Unknown (SQP-1): The description triggers on "MCP or REST access, readiness, authentication, conversations, and grounded questions" — ter (`SKILL.md:3`)

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
