# Skill Benchmark: portfolio-optimization

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `portfolio-optimization`
- Evaluation date: 2026-09-14
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (2 positive, 2 negative)
- Dataset digest: `sha256:97b53fc5f0fc9cbbb6e594394ff89d950c9b5b713c0f4a6594c968dade42869f` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 74.4% — baseline ran, but no comparable score was available; uplift unavailable | 72.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 62.5% → 75.0% (+12.5 points) | 87.5% → 100.0% (+12.5 points) |
| Correctness | 55.0% → 75.0% (+20.0 points) | 47.5% → 45.0% (-2.5 points) |
| Discoverability | 90.0% — baseline ran, but no comparable score was available; uplift unavailable | 87.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 38.5% → 51.2% (+12.7 points) | 25.2% → 39.3% (+14.1 points) |
| Efficiency | 80.8% — baseline ran, but no comparable score was available; uplift unavailable | 90.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,418,932 | 7,517,057 | N/A | N/A | skill 4/4; base 8/8 |
| claude-code | build-optimal-cvar | 342,279 | 1,704,486 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | efficient-frontier-plot | 391,118 | 4,233,182 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | neg-nn-price-forecast | 2,126,146 | 976,910 | +1,149,236 | +117.64% | skill 1/1; base 1/1 |
| claude-code | neg-vehicle-routing | 559,389 | 602,479 | -43,090 | -7.15% | skill 1/1; base 1/1 |
| codex | All cases | 505,004 | 7,533,869 | N/A | N/A | skill 4/4; base 8/8 |
| codex | build-optimal-cvar | 101,131 | 2,011,116 | N/A | N/A | skill 1/1; base 3/3 |
| codex | efficient-frontier-plot | 304,474 | 5,495,535 | N/A | N/A | skill 1/1; base 3/3 |
| codex | neg-nn-price-forecast | 71,955 | 13,737 | +58,218 | +423.80% | skill 1/1; base 1/1 |
| codex | neg-vehicle-routing | 27,444 | 13,481 | +13,963 | +103.58% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 3,923,936 | 15,050,926 | N/A | N/A | skill 8/8; base 16/16 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 4 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/portfolio-optimization/SKILL.md`)
- **LOW** QUALITY/quality_reliability: Inputs are used but no dedicated Inputs section is documented (`skills/portfolio-optimization/SKILL.md`)
- **LOW** SECURITY/Unknown (SQP-1): The trigger list in L041 includes several generic phrases such as "compare allocations", "construct an allocation", "ass (`SKILL.md:41`)
- **LOW** SECURITY/Unknown (SQP-2): The `rebalance_monthly` function writes price data to a hardcoded `/tmp/portfolio_optimization_rebalance_prices.csv` pat (`references/workflows/agent_recipes.md:300`)

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
