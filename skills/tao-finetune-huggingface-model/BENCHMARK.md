# Skill Benchmark: tao-finetune-huggingface-model

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `tao-finetune-huggingface-model`
- Evaluation date: 2026-09-21
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (4 positive)
- Dataset digest: `sha256:28fd5bf04229375b8bfea649e9cb787a04bb482b80b6ae2ae31e5ff9f076838a` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 80.2% — baseline ran, but no comparable score was available; uplift unavailable | 58.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 25.7% → 85.0% (+59.3 points) | 52.0% → 55.0% (+3.0 points) |
| Discoverability | 58.8% — baseline ran, but no comparable score was available; uplift unavailable | 0.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 35.2% → 81.7% (+46.5 points) | 45.0% → 39.1% (-5.9 points) |
| Efficiency | 75.7% — baseline ran, but no comparable score was available; uplift unavailable | 98.8% → 99.3% (+0.5 points) |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 594,721 | 755,784 | N/A | N/A | skill 4/4; base 7/7 |
| claude-code | tao-finetune-huggingface-model-basic | 70,212 | 506,739 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-finetune-huggingface-model-dedicated-routing | 206,194 | 31,329 | +174,865 | +558.16% | skill 1/1; base 1/1 |
| claude-code | tao-finetune-huggingface-model-dedicated-routing-non-cosmos | 288,161 | 61,641 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | tao-finetune-huggingface-model-unclaimed-routing | 30,154 | 156,075 | -125,921 | -80.68% | skill 1/1; base 1/1 |
| codex | All cases | 111,184 | 76,963 | N/A | N/A | skill 8/8; base 5/5 |
| codex | tao-finetune-huggingface-model-basic | 13,984 | 13,606 | +378 | +2.78% | skill 1/1; base 1/1 |
| codex | tao-finetune-huggingface-model-dedicated-routing | 41,788 | 31,641 | N/A | N/A | skill 3/3; base 2/2 |
| codex | tao-finetune-huggingface-model-dedicated-routing-non-cosmos | 41,565 | 17,862 | N/A | N/A | skill 3/3; base 1/1 |
| codex | tao-finetune-huggingface-model-unclaimed-routing | 13,847 | 13,854 | -7 | -0.05% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 705,905 | 832,747 | -126,842 | -15.23% | skill 12/12; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 63 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **CRITICAL** CONTENT_DEDUP/chunk_count_limit: Tier 2 produced more than 512 content chunks. (`references/workflow-intake-preflight.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/applications/tao-finetune-huggingface-model/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in docker-runs.md (`skills/applications/tao-finetune-huggingface-model/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'tags' is ignored; use 'metadata.tags' (`skills/applications/tao-finetune-huggingface-model/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/applications/tao-finetune-huggingface-model`)
- 59 additional finding(s) are available in the full evaluation artifacts.

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
