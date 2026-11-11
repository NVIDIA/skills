# Skill Benchmark: tao-finetune-cosmos-reason

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `tao-finetune-cosmos-reason`
- Evaluation date: 2026-10-16
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 7 evaluation tasks (7 positive)
- Dataset digest: `sha256:81bc83868a378e3f3fdece654f6a44dce369888b12a2f872175a72b84b090797` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 90.1% — baseline ran, but no comparable score was available; uplift unavailable | 81.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 40.0% → 85.7% (+45.7 points) | 54.3% → 85.7% (+31.4 points) |
| Discoverability | 97.1% — baseline ran, but no comparable score was available; uplift unavailable | 61.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 30.3% → 80.3% (+50.0 points) | 33.8% → 61.0% (+27.2 points) |
| Efficiency | 87.1% — baseline ran, but no comparable score was available; uplift unavailable | 98.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,855,578 | 1,391,839 | +1,463,739 | +105.17% | skill 7/7; base 7/7 |
| claude-code | tao-finetune-cosmos-reason-backend-selection | 382,351 | 130,202 | +252,149 | +193.66% | skill 1/1; base 1/1 |
| claude-code | tao-finetune-cosmos-reason-basic | 259,189 | 411,524 | -152,335 | -37.02% | skill 1/1; base 1/1 |
| claude-code | tao-finetune-cosmos-reason-container-runtime | 67,615 | 91,954 | -24,339 | -26.47% | skill 1/1; base 1/1 |
| claude-code | tao-finetune-cosmos-reason-conversation-single-gpu | 389,842 | 284,630 | +105,212 | +36.96% | skill 1/1; base 1/1 |
| claude-code | tao-finetune-cosmos-reason-dense-sft-parity | 925,629 | 193,433 | +732,196 | +378.53% | skill 1/1; base 1/1 |
| claude-code | tao-finetune-cosmos-reason-evaluation-inheritance | 588,069 | 186,676 | +401,393 | +215.02% | skill 1/1; base 1/1 |
| claude-code | tao-finetune-cosmos-reason-framework-status | 242,883 | 93,420 | +149,463 | +159.99% | skill 1/1; base 1/1 |
| codex | All cases | 457,091 | 190,564 | +266,527 | +139.86% | skill 7/7; base 7/7 |
| codex | tao-finetune-cosmos-reason-backend-selection | 14,496 | 25,656 | -11,160 | -43.50% | skill 1/1; base 1/1 |
| codex | tao-finetune-cosmos-reason-basic | 84,751 | 14,019 | +70,732 | +504.54% | skill 1/1; base 1/1 |
| codex | tao-finetune-cosmos-reason-container-runtime | 30,503 | 13,792 | +16,711 | +121.16% | skill 1/1; base 1/1 |
| codex | tao-finetune-cosmos-reason-conversation-single-gpu | 192,087 | 43,503 | +148,584 | +341.55% | skill 1/1; base 1/1 |
| codex | tao-finetune-cosmos-reason-dense-sft-parity | 49,714 | 15,664 | +34,050 | +217.38% | skill 1/1; base 1/1 |
| codex | tao-finetune-cosmos-reason-evaluation-inheritance | 70,793 | 14,239 | +56,554 | +397.18% | skill 1/1; base 1/1 |
| codex | tao-finetune-cosmos-reason-framework-status | 14,747 | 63,691 | -48,944 | -76.85% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 3,312,669 | 1,582,403 | +1,730,266 | +109.34% | skill 14/14; base 14/14 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 96 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 28 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 7 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/cosmos-actions-parameters.md and references/cosmos-reason-parameters.md:
  "## Error Patterns" in references/cosmos-actions-parameters.md (lines 219-222)
  vs "## Error Patterns" in references/cosmos-reason-parameters.md (lines 135-138) (`references/cosmos-actions-parameters.md:219`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/cosmos-actions-parameters.md and references/cosmos-reason-parameters.md:
  "### Logging" in references/cosmos-actions-parameters.md (lines 203-206)
  vs "### Logging" in references/cosmos-reason-parameters.md (lines 110-113) (`references/cosmos-actions-parameters.md:203`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/cosmos-actions-parameters.md and references/cosmos-reason-parameters.md:
  "## Error Patterns" in references/cosmos-actions-parameters.md (lines 218-218)
  vs "## Error Patterns" in references/cosmos-reason-parameters.md (lines 134-134) (`references/cosmos-actions-parameters.md:218`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/cosmos-actions-parameters.md and references/cosmos-reason-parameters.md:
  "## Error Patterns" in references/cosmos-actions-parameters.md (lines 215-215)
  vs "## Error Patterns" in references/cosmos-reason-parameters.md (lines 131-131) (`references/cosmos-actions-parameters.md:215`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/cosmos-actions-parameters.md and references/cosmos-reason-parameters.md:
  "## Error Patterns" in references/cosmos-actions-parameters.md (lines 217-217)
  vs "## Error Patterns" in references/cosmos-reason-parameters.md (lines 133-133) (`references/cosmos-actions-parameters.md:217`)
- 119 additional finding(s) are available in the full evaluation artifacts.

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
