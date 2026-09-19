# Skill Benchmark: tao-finetune-cosmos-reason

> ❌ **Overall verdict: FAIL — Publication blocked**

The skill should be reviewed before publication. Address the blocking findings below, then rerun Skill Evaluator.

## Evaluation Metadata

- Skill: `tao-finetune-cosmos-reason`
- Evaluation date: 2026-09-18
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
| Overall | 86.1% — baseline ran, but no comparable score was available; uplift unavailable | 74.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 96.7% → 100.0% (+3.3 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 26.7% → 68.6% (+41.9 points) | 45.3% → 84.0% (+38.7 points) |
| Discoverability | 96.4% — baseline ran, but no comparable score was available; uplift unavailable | 35.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 20.7% → 80.0% (+59.3 points) | 26.9% → 52.6% (+25.7 points) |
| Efficiency | 85.6% — baseline ran, but no comparable score was available; uplift unavailable | 98.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,990,421 | 4,090,642 | N/A | N/A | skill 7/7; base 15/15 |
| claude-code | tao-finetune-cosmos-reason-backend-selection | 254,087 | 399,349 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-finetune-cosmos-reason-basic | 110,166 | 1,016,263 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-finetune-cosmos-reason-container-runtime | 116,327 | 91,669 | +24,658 | +26.90% | skill 1/1; base 1/1 |
| claude-code | tao-finetune-cosmos-reason-conversation-single-gpu | 305,827 | 1,597,430 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | tao-finetune-cosmos-reason-dense-sft-parity | 775,073 | 601,218 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-finetune-cosmos-reason-evaluation-inheritance | 215,574 | 163,831 | +51,743 | +31.58% | skill 1/1; base 1/1 |
| claude-code | tao-finetune-cosmos-reason-framework-status | 213,367 | 220,882 | N/A | N/A | skill 1/1; base 2/2 |
| codex | All cases | 465,079 | 433,088 | N/A | N/A | skill 10/10; base 15/15 |
| codex | tao-finetune-cosmos-reason-backend-selection | 49,239 | 61,184 | N/A | N/A | skill 1/1; base 3/3 |
| codex | tao-finetune-cosmos-reason-basic | 13,969 | 41,915 | N/A | N/A | skill 1/1; base 3/3 |
| codex | tao-finetune-cosmos-reason-container-runtime | 14,078 | 18,955 | -4,877 | -25.73% | skill 1/1; base 1/1 |
| codex | tao-finetune-cosmos-reason-conversation-single-gpu | 249,200 | 46,380 | +202,820 | +437.30% | skill 1/1; base 1/1 |
| codex | tao-finetune-cosmos-reason-dense-sft-parity | 45,777 | 83,360 | N/A | N/A | skill 2/2; base 3/3 |
| codex | tao-finetune-cosmos-reason-evaluation-inheritance | 49,807 | 14,376 | +35,431 | +246.46% | skill 1/1; base 1/1 |
| codex | tao-finetune-cosmos-reason-framework-status | 43,009 | 166,918 | -123,909 | -74.23% | skill 3/3; base 3/3 |
| ALL AGENTS | Dataset aggregate | 2,455,500 | 4,523,730 | N/A | N/A | skill 17/17; base 30/30 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **FAILED** | 11 validator(s); 97 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 28 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 7 task(s) |

## Blocking Findings

- **MEDIUM** BANDIT/B310:blacklist: Audit url open for permitted schemes. Allowing use of file:/ or custom schemes is often unexpected. (CWE-22) (`skills/models/tao-finetune-cosmos-reason/scripts/cosmos_workflow.py:158`)

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **CRITICAL** CONTENT_DEDUP/llm_error: LLM analysis failed for a content cluster (`skills/models/tao-finetune-cosmos-reason`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/cosmos-actions-parameters.md and references/cosmos-reason-parameters.md:
  "### Logging" in references/cosmos-actions-parameters.md (lines 203-206)
  vs "### Logging" in references/cosmos-reason-parameters.md (lines 110-113) (`references/cosmos-actions-parameters.md:203`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/cosmos-actions-parameters.md and references/cosmos-reason-parameters.md:
  "## Error Patterns" in references/cosmos-actions-parameters.md (lines 215-215)
  vs "## Error Patterns" in references/cosmos-reason-parameters.md (lines 130-130) (`references/cosmos-actions-parameters.md:215`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/cosmos-actions-parameters.md and references/cosmos-reason-parameters.md:
  "## Error Patterns" in references/cosmos-actions-parameters.md (lines 218-218)
  vs "## Error Patterns" in references/cosmos-reason-parameters.md (lines 133-133) (`references/cosmos-actions-parameters.md:218`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/cosmos-actions-parameters.md and references/cosmos-reason-parameters.md:
  "## Error Patterns" in references/cosmos-actions-parameters.md (lines 217-217)
  vs "## Error Patterns" in references/cosmos-reason-parameters.md (lines 132-132) (`references/cosmos-actions-parameters.md:217`)
- 120 additional finding(s) are available in the full evaluation artifacts.

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
