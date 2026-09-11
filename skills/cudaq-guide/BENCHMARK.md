# Skill Benchmark: cudaq-guide

> ⚠️ **Overall verdict: INCOMPLETE — Required evidence is missing**

One or more required evaluation tiers did not complete, so this benchmark is not publication-complete.

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
| Overall | 94.9% — baseline ran, but no comparable score was available; uplift unavailable | 88.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 84.6% → 92.3% (+7.7 points) |
| Correctness | 87.7% → 100.0% (+12.3 points) | 92.3% → 95.4% (+3.1 points) |
| Discoverability | 97.5% — baseline ran, but no comparable score was available; uplift unavailable | 93.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 67.1% → 85.4% (+18.3 points) | 73.9% → 84.8% (+10.9 points) |
| Efficiency | 91.6% — baseline ran, but no comparable score was available; uplift unavailable | 75.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,634,303 | 942,384 | +691,919 | +73.42% | skill 13/13; base 13/13 |
| claude-code | cudaq-guide-applications-001 | 96,349 | 30,148 | +66,201 | +219.59% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-author-adjoint-001 | 103,383 | 93,926 | +9,457 | +10.07% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-author-api-001 | 102,103 | 30,330 | +71,773 | +236.64% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-author-constraint-001 | 102,405 | 31,998 | +70,407 | +220.04% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-gpu-sim-001 | 96,409 | 31,135 | +65,274 | +209.65% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-install-001 | 96,753 | 30,946 | +65,807 | +212.65% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-neg-001 | 29,140 | 29,107 | +33 | +0.11% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-neg-002 | 29,620 | 29,556 | +64 | +0.22% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-neg-003 | 129,808 | 124,730 | +5,078 | +4.07% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-parallelize-001 | 96,685 | 31,982 | +64,703 | +202.31% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-qpu-001 | 172,547 | 61,872 | +110,675 | +178.88% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-route-qiskit-001 | 171,587 | 91,745 | +79,842 | +87.03% | skill 1/1; base 1/1 |
| claude-code | cudaq-guide-test-program-001 | 407,514 | 324,909 | +82,605 | +25.42% | skill 1/1; base 1/1 |
| codex | All cases | 1,438,443 | 1,067,607 | +370,836 | +34.74% | skill 13/13; base 13/13 |
| codex | cudaq-guide-applications-001 | 54,575 | 24,620 | +29,955 | +121.67% | skill 1/1; base 1/1 |
| codex | cudaq-guide-author-adjoint-001 | 53,727 | 46,238 | +7,489 | +16.20% | skill 1/1; base 1/1 |
| codex | cudaq-guide-author-api-001 | 50,488 | 17,820 | +32,668 | +183.32% | skill 1/1; base 1/1 |
| codex | cudaq-guide-author-constraint-001 | 56,999 | 90,531 | -33,532 | -37.04% | skill 1/1; base 1/1 |
| codex | cudaq-guide-gpu-sim-001 | 62,992 | 17,858 | +45,134 | +252.74% | skill 1/1; base 1/1 |
| codex | cudaq-guide-install-001 | 62,709 | 44,392 | +18,317 | +41.26% | skill 1/1; base 1/1 |
| codex | cudaq-guide-neg-001 | 13,274 | 13,242 | +32 | +0.24% | skill 1/1; base 1/1 |
| codex | cudaq-guide-neg-002 | 13,355 | 13,326 | +29 | +0.22% | skill 1/1; base 1/1 |
| codex | cudaq-guide-neg-003 | 402,870 | 431,387 | -28,517 | -6.61% | skill 1/1; base 1/1 |
| codex | cudaq-guide-parallelize-001 | 73,773 | 20,357 | +53,416 | +262.40% | skill 1/1; base 1/1 |
| codex | cudaq-guide-qpu-001 | 92,086 | 28,452 | +63,634 | +223.65% | skill 1/1; base 1/1 |
| codex | cudaq-guide-route-qiskit-001 | 28,555 | 32,007 | -3,452 | -10.79% | skill 1/1; base 1/1 |
| codex | cudaq-guide-test-program-001 | 473,040 | 287,377 | +185,663 | +64.61% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 3,072,746 | 2,009,991 | +1,062,755 | +52.87% | skill 26/26; base 26/26 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 1 validator(s); 4 finding(s) |
| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |
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
