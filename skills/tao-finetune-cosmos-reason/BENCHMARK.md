# Skill Benchmark: tao-finetune-cosmos-reason

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `tao-finetune-cosmos-reason`
- Evaluation date: 2026-09-21
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 7 evaluation tasks (7 positive)
- Dataset digest: `sha256:81bc83868a378e3f3fdece654f6a44dce369888b12a2f872175a72b84b090797` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 87.8% — baseline ran, but no comparable score was available; uplift unavailable | 72.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 94.1% → 100.0% (+5.9 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 29.4% → 74.3% (+44.9 points) | 46.7% → 77.8% (+31.1 points) |
| Discoverability | 94.3% — baseline ran, but no comparable score was available; uplift unavailable | 40.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 17.8% → 85.4% (+67.6 points) | 26.7% → 44.1% (+17.4 points) |
| Efficiency | 84.9% — baseline ran, but no comparable score was available; uplift unavailable | 98.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,225,288 | 3,497,765 | N/A | N/A | skill 7/7; base 17/17 |
| claude-code | tao-finetune-cosmos-reason-backend-selection | 255,458 | 451,656 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-finetune-cosmos-reason-basic | 111,670 | 951,931 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-finetune-cosmos-reason-container-runtime | 176,711 | 123,467 | +53,244 | +43.12% | skill 1/1; base 1/1 |
| claude-code | tao-finetune-cosmos-reason-conversation-single-gpu | 654,692 | 848,907 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-finetune-cosmos-reason-dense-sft-parity | 660,079 | 707,646 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-finetune-cosmos-reason-evaluation-inheritance | 117,562 | 127,972 | -10,410 | -8.13% | skill 1/1; base 1/1 |
| claude-code | tao-finetune-cosmos-reason-framework-status | 249,116 | 286,186 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 474,847 | 312,687 | N/A | N/A | skill 9/9; base 12/12 |
| codex | tao-finetune-cosmos-reason-backend-selection | 31,154 | 78,329 | N/A | N/A | skill 1/1; base 3/3 |
| codex | tao-finetune-cosmos-reason-basic | 14,108 | 41,912 | N/A | N/A | skill 1/1; base 3/3 |
| codex | tao-finetune-cosmos-reason-container-runtime | 13,944 | 32,260 | -18,316 | -56.78% | skill 1/1; base 1/1 |
| codex | tao-finetune-cosmos-reason-conversation-single-gpu | 99,769 | 72,574 | +27,195 | +37.47% | skill 1/1; base 1/1 |
| codex | tao-finetune-cosmos-reason-dense-sft-parity | 30,850 | 14,950 | +15,900 | +106.35% | skill 1/1; base 1/1 |
| codex | tao-finetune-cosmos-reason-evaluation-inheritance | 240,999 | 14,475 | +226,524 | +1564.93% | skill 1/1; base 1/1 |
| codex | tao-finetune-cosmos-reason-framework-status | 44,023 | 58,187 | N/A | N/A | skill 3/3; base 2/2 |
| ALL AGENTS | Dataset aggregate | 2,700,135 | 3,810,452 | N/A | N/A | skill 16/16; base 29/29 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 98 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 28 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 7 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/cosmos-actions-parameters.md and references/cosmos-reason-parameters.md:
  "## Error Patterns" in references/cosmos-actions-parameters.md (lines 218-218)
  vs "## Error Patterns" in references/cosmos-reason-parameters.md (lines 133-133) (`references/cosmos-actions-parameters.md:218`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/cosmos-actions-parameters.md and references/cosmos-reason-parameters.md:
  "### Logging" in references/cosmos-actions-parameters.md (lines 203-206)
  vs "### Logging" in references/cosmos-reason-parameters.md (lines 110-113) (`references/cosmos-actions-parameters.md:203`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/cosmos-actions-parameters.md and references/cosmos-reason-parameters.md:
  "## Error Patterns" in references/cosmos-actions-parameters.md (lines 219-222)
  vs "## Error Patterns" in references/cosmos-reason-parameters.md (lines 134-137) (`references/cosmos-actions-parameters.md:219`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/cosmos-actions-parameters.md and references/cosmos-reason-parameters.md:
  "## Error Patterns" in references/cosmos-actions-parameters.md (lines 217-217)
  vs "## Error Patterns" in references/cosmos-reason-parameters.md (lines 132-132) (`references/cosmos-actions-parameters.md:217`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/cosmos-actions-parameters.md and references/cosmos-reason-parameters.md:
  "## Error Patterns" in references/cosmos-actions-parameters.md (lines 215-215)
  vs "## Error Patterns" in references/cosmos-reason-parameters.md (lines 130-130) (`references/cosmos-actions-parameters.md:215`)
- 121 additional finding(s) are available in the full evaluation artifacts.

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
