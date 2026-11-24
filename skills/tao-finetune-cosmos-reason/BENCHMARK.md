# Skill Benchmark: tao-finetune-cosmos-reason

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `tao-finetune-cosmos-reason`
- Validation status: `passed`
- Evaluation date: 2026-11-20
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 7 evaluation tasks (7 positive)
- Dataset digest: `sha256:81bc83868a378e3f3fdece654f6a44dce369888b12a2f872175a72b84b090797` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| tao-finetune-cosmos-reason | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| tao-finetune-cosmos-reason | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| tao-finetune-cosmos-reason | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| tao-finetune-cosmos-reason | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| tao-finetune-cosmos-reason | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `tao-finetune-cosmos-reason`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 86.4% — baseline ran, but no comparable score was available; uplift unavailable | 75.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 42.9% → 91.4% (+48.5 points) | 57.1% → 82.9% (+25.8 points) |
| Discoverability | 90.0% — baseline ran, but no comparable score was available; uplift unavailable | 62.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 19.1% → 62.0% (+42.9 points) | 17.1% → 39.0% (+21.9 points) |
| Efficiency | 88.7% — baseline ran, but no comparable score was available; uplift unavailable | 95.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.8% → 92.8% (-5.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 86.8% → 90.7% (+3.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 910,542 | 724,852 | +185,690 | +25.62% | skill 7/7; base 7/7 |
| claude-code | tao-finetune-cosmos-reason-backend-selection | 80,681 | 109,796 | -29,115 | -26.52% | skill 1/1; base 1/1 |
| claude-code | tao-finetune-cosmos-reason-basic | 80,504 | 141,729 | -61,225 | -43.20% | skill 1/1; base 1/1 |
| claude-code | tao-finetune-cosmos-reason-container-runtime | 79,531 | 71,142 | +8,389 | +11.79% | skill 1/1; base 1/1 |
| claude-code | tao-finetune-cosmos-reason-conversation-single-gpu | 242,669 | 147,094 | +95,575 | +64.98% | skill 1/1; base 1/1 |
| claude-code | tao-finetune-cosmos-reason-dense-sft-parity | 125,055 | 109,802 | +15,253 | +13.89% | skill 1/1; base 1/1 |
| claude-code | tao-finetune-cosmos-reason-evaluation-inheritance | 129,320 | 72,546 | +56,774 | +78.26% | skill 1/1; base 1/1 |
| claude-code | tao-finetune-cosmos-reason-framework-status | 172,782 | 72,743 | +100,039 | +137.52% | skill 1/1; base 1/1 |
| codex | All cases | 472,284 | 228,647 | +243,637 | +106.56% | skill 7/7; base 7/7 |
| codex | tao-finetune-cosmos-reason-backend-selection | 71,517 | 72,785 | -1,268 | -1.74% | skill 1/1; base 1/1 |
| codex | tao-finetune-cosmos-reason-basic | 13,608 | 13,371 | +237 | +1.77% | skill 1/1; base 1/1 |
| codex | tao-finetune-cosmos-reason-container-runtime | 47,524 | 13,438 | +34,086 | +253.65% | skill 1/1; base 1/1 |
| codex | tao-finetune-cosmos-reason-conversation-single-gpu | 182,869 | 62,453 | +120,416 | +192.81% | skill 1/1; base 1/1 |
| codex | tao-finetune-cosmos-reason-dense-sft-parity | 30,372 | 38,789 | -8,417 | -21.70% | skill 1/1; base 1/1 |
| codex | tao-finetune-cosmos-reason-evaluation-inheritance | 112,429 | 13,957 | +98,472 | +705.54% | skill 1/1; base 1/1 |
| codex | tao-finetune-cosmos-reason-framework-status | 13,965 | 13,854 | +111 | +0.80% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,382,826 | 953,499 | +429,327 | +45.03% | skill 14/14; base 14/14 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 95 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 31 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 7 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/cosmos-actions-parameters.md and references/cosmos-reason-parameters.md:
  "## Error Patterns" in references/cosmos-actions-parameters.md (lines 218-218)
  vs "## Error Patterns" in references/cosmos-reason-parameters.md (lines 134-134) (`references/cosmos-actions-parameters.md:218`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/cosmos-actions-parameters.md and references/cosmos-reason-parameters.md:
  "## Error Patterns" in references/cosmos-actions-parameters.md (lines 219-222)
  vs "## Error Patterns" in references/cosmos-reason-parameters.md (lines 135-138) (`references/cosmos-actions-parameters.md:219`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/cosmos-actions-parameters.md and references/cosmos-reason-parameters.md:
  "## Error Patterns" in references/cosmos-actions-parameters.md (lines 215-215)
  vs "## Error Patterns" in references/cosmos-reason-parameters.md (lines 131-131) (`references/cosmos-actions-parameters.md:215`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/cosmos-actions-parameters.md and references/cosmos-reason-parameters.md:
  "## Error Patterns" in references/cosmos-actions-parameters.md (lines 217-217)
  vs "## Error Patterns" in references/cosmos-reason-parameters.md (lines 133-133) (`references/cosmos-actions-parameters.md:217`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/cosmos-actions-parameters.md and references/cosmos-reason-parameters.md:
  "### Logging" in references/cosmos-actions-parameters.md (lines 203-206)
  vs "### Logging" in references/cosmos-reason-parameters.md (lines 110-113) (`references/cosmos-actions-parameters.md:203`)
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
