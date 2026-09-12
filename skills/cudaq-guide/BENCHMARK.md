# Skill Benchmark: cudaq-guide

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cudaq-guide`
- Evaluation date: 2026-09-11
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 13 evaluation tasks (10 positive, 3 negative)
- Dataset digest: `sha256:7884d2878ed008a3969a7668a02fb2f9fda8e70495368a69ca8bf4ce1f6071dd` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 93.6% — baseline ran, but no comparable score was available; uplift unavailable | 89.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 85.7% → 92.3% (+6.6 points) |
| Correctness | 92.3% → 98.5% (+6.2 points) | 90.0% → 98.5% (+8.5 points) |
| Discoverability | 97.5% — baseline ran, but no comparable score was available; uplift unavailable | 92.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 69.2% → 79.9% (+10.7 points) | 66.0% → 86.1% (+20.1 points) |
| Efficiency | 91.9% — baseline ran, but no comparable score was available; uplift unavailable | 77.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,666,118 | 1,251,598 | +414,520 | +33.12% | skill 13/13; base 13/13 |
| claude-code | cudaq-guide-applications-001 | 96,575 | 30,095 | +66,480 | +220.90% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-author-adjoint-001 | 103,562 | 65,498 | +38,064 | +58.11% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-author-api-001 | 102,254 | 30,416 | +71,838 | +236.18% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-author-constraint-001 | 102,286 | 133,926 | -31,640 | -23.62% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-gpu-sim-001 | 96,542 | 31,244 | +65,298 | +208.99% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-install-001 | 96,783 | 31,099 | +65,684 | +211.21% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-neg-001 | 29,099 | 29,092 | +7 | +0.02% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-neg-002 | 29,417 | 29,600 | -183 | -0.62% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-neg-003 | 155,890 | 191,018 | -35,128 | -18.39% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-parallelize-001 | 96,696 | 31,451 | +65,245 | +207.45% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-qpu-001 | 171,868 | 162,627 | +9,241 | +5.68% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-route-qiskit-001 | 137,822 | 124,365 | +13,457 | +10.82% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-test-program-001 | 447,324 | 361,167 | +86,157 | +23.86% | skill 1/1; base 1/1 |
| codex | All cases | 1,861,622 | 1,290,210 | N/A | N/A | skill 13/13; base 14/14 |
| codex | cudaq-guide-applications-001 | 59,610 | 32,305 | +27,305 | +84.52% | skill 1/1; base 1/1 |
| codex | cudaq-guide-author-adjoint-001 | 64,364 | 94,709 | N/A | N/A | skill 1/1; base 2/2 |
| codex | cudaq-guide-author-api-001 | 45,999 | 24,417 | +21,582 | +88.39% | skill 1/1; base 1/1 |
| codex | cudaq-guide-author-constraint-001 | 46,233 | 44,952 | +1,281 | +2.85% | skill 1/1; base 1/1 |
| codex | cudaq-guide-gpu-sim-001 | 78,984 | 17,935 | +61,049 | +340.39% | skill 1/1; base 1/1 |
| codex | cudaq-guide-install-001 | 75,783 | 37,171 | +38,612 | +103.88% | skill 1/1; base 1/1 |
| codex | cudaq-guide-neg-001 | 13,272 | 13,234 | +38 | +0.29% | skill 1/1; base 1/1 |
| codex | cudaq-guide-neg-002 | 13,380 | 13,308 | +72 | +0.54% | skill 1/1; base 1/1 |
| codex | cudaq-guide-neg-003 | 532,599 | 598,297 | -65,698 | -10.98% | skill 1/1; base 1/1 |
| codex | cudaq-guide-parallelize-001 | 50,307 | 25,488 | +24,819 | +97.38% | skill 1/1; base 1/1 |
| codex | cudaq-guide-qpu-001 | 87,178 | 46,231 | +40,947 | +88.57% | skill 1/1; base 1/1 |
| codex | cudaq-guide-route-qiskit-001 | 46,712 | 44,623 | +2,089 | +4.68% | skill 1/1; base 1/1 |
| codex | cudaq-guide-test-program-001 | 747,201 | 297,540 | +449,661 | +151.13% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 3,527,740 | 2,541,808 | N/A | N/A | skill 26/26; base 27/27 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 4 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 13 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'author' is ignored; use 'metadata.author' (`skills/cudaq-guide/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'tags' is ignored; use 'metadata.tags' (`skills/cudaq-guide/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/cudaq-guide/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/cudaq-guide/SKILL.md`)

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
