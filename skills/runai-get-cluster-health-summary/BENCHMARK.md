# Skill Benchmark: runai-get-cluster-health-summary

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `runai-get-cluster-health-summary`
- Validation status: `passed`
- Evaluation date: 2026-10-28
- Evaluator version: `1.6.2`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 22 evaluation tasks (15 positive, 7 negative)
- Dataset digest: `sha256:61933beeca043b8c5e2005db58c885512ae496d6b49d9af0571e0233c16a5f23` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| runai-get-cluster-health-summary | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| runai-get-cluster-health-summary | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| runai-get-cluster-health-summary | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| runai-get-cluster-health-summary | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| runai-get-cluster-health-summary | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |

- Controls for `runai-get-cluster-health-summary`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 89.7% — baseline ran, but no comparable score was available; uplift unavailable | 85.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 90.9% → 100.0% (+9.1 points) |
| Correctness | 79.1% → 88.2% (+9.1 points) | 76.4% → 91.8% (+15.4 points) |
| Discoverability | 88.0% — baseline ran, but no comparable score was available; uplift unavailable | 91.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 60.8% → 77.2% (+16.4 points) | 55.4% → 83.6% (+28.2 points) |
| Efficiency | 94.8% — baseline ran, but no comparable score was available; uplift unavailable | 59.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.3% → 93.0% (-4.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 92.1% → 89.8% (-2.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,380,093 | 2,714,239 | +1,665,854 | +61.37% | skill 22/22; base 22/22 |
| claude-code | A1-lab-cluster-verdict-per-pool | 351,689 | 118,765 | +232,924 | +196.12% | skill 1/1; base 1/1 |
| claude-code | A10-point-in-time-after-node-drop | 352,829 | 77,064 | +275,765 | +357.84% | skill 1/1; base 1/1 |
| claude-code | A2-live-cluster-healthy | 131,836 | 78,561 | +53,275 | +67.81% | skill 1/1; base 1/1 |
| claude-code | A3-scheduler-stuck-or-full | 302,655 | 172,252 | +130,403 | +75.70% | skill 1/1; base 1/1 |
| claude-code | A4-nodes-down-gpus-offline | 352,162 | 76,881 | +275,281 | +358.06% | skill 1/1; base 1/1 |
| claude-code | A5-pool-not-scheduling | 234,022 | 248,836 | -14,814 | -5.95% | skill 1/1; base 1/1 |
| claude-code | A6-edge-pool-offline | 183,683 | 77,895 | +105,788 | +135.81% | skill 1/1; base 1/1 |
| claude-code | A7-allocation-and-utilization | 352,742 | 125,288 | +227,454 | +181.54% | skill 1/1; base 1/1 |
| claude-code | B1-quota-fairness-routes-out | 163,569 | 202,450 | -38,881 | -19.21% | skill 1/1; base 1/1 |
| claude-code | B2-idle-gpus-routes-out | 79,172 | 78,497 | +675 | +0.86% | skill 1/1; base 1/1 |
| claude-code | B3-one-workload-routes-out | 200,522 | 196,242 | +4,280 | +2.18% | skill 1/1; base 1/1 |
| claude-code | B4-node-issue-routes-out | 161,881 | 160,574 | +1,307 | +0.81% | skill 1/1; base 1/1 |
| claude-code | C1-no-cluster-id-asks | 128,932 | 116,374 | +12,558 | +10.79% | skill 1/1; base 1/1 |
| claude-code | C2-full-pool-is-capacity-not-stuck | 239,005 | 169,576 | +69,429 | +40.94% | skill 1/1; base 1/1 |
| claude-code | C3-free-gpus-but-spec-pending | 347,104 | 122,758 | +224,346 | +182.75% | skill 1/1; base 1/1 |
| claude-code | C4-missing-wait-time-metric | 116,295 | 115,568 | +727 | +0.63% | skill 1/1; base 1/1 |
| claude-code | C5-cpu-node-not-gpu-loss | 77,395 | 76,954 | +441 | +0.57% | skill 1/1; base 1/1 |
| claude-code | C7-ready-node-gpu-blind-spot | 178,817 | 76,779 | +102,038 | +132.90% | skill 1/1; base 1/1 |
| claude-code | C8-no-kubectl-fallback | 115,266 | 114,534 | +732 | +0.64% | skill 1/1; base 1/1 |
| claude-code | D1-history-out-of-scope | 79,522 | 78,912 | +610 | +0.77% | skill 1/1; base 1/1 |
| claude-code | D2-cordon-write | 115,263 | 114,504 | +759 | +0.66% | skill 1/1; base 1/1 |
| claude-code | D6-unrelated-request | 115,732 | 114,975 | +757 | +0.66% | skill 1/1; base 1/1 |
| codex | All cases | 4,176,807 | 2,092,474 | +2,084,333 | +99.61% | skill 22/22; base 22/22 |
| codex | A1-lab-cluster-verdict-per-pool | 370,995 | 46,019 | +324,976 | +706.18% | skill 1/1; base 1/1 |
| codex | A10-point-in-time-after-node-drop | 371,890 | 45,680 | +326,210 | +714.12% | skill 1/1; base 1/1 |
| codex | A2-live-cluster-healthy | 157,704 | 75,863 | +81,841 | +107.88% | skill 1/1; base 1/1 |
| codex | A3-scheduler-stuck-or-full | 369,853 | 335,546 | +34,307 | +10.22% | skill 1/1; base 1/1 |
| codex | A4-nodes-down-gpus-offline | 366,271 | 30,059 | +336,212 | +1118.51% | skill 1/1; base 1/1 |
| codex | A5-pool-not-scheduling | 303,631 | 219,887 | +83,744 | +38.09% | skill 1/1; base 1/1 |
| codex | A6-edge-pool-offline | 303,918 | 76,704 | +227,214 | +296.22% | skill 1/1; base 1/1 |
| codex | A7-allocation-and-utilization | 369,941 | 111,472 | +258,469 | +231.87% | skill 1/1; base 1/1 |
| codex | B1-quota-fairness-routes-out | 127,001 | 108,413 | +18,588 | +17.15% | skill 1/1; base 1/1 |
| codex | B2-idle-gpus-routes-out | 48,927 | 63,326 | -14,399 | -22.74% | skill 1/1; base 1/1 |
| codex | B3-one-workload-routes-out | 90,414 | 105,307 | -14,893 | -14.14% | skill 1/1; base 1/1 |
| codex | B4-node-issue-routes-out | 166,049 | 121,324 | +44,725 | +36.86% | skill 1/1; base 1/1 |
| codex | C1-no-cluster-id-asks | 72,952 | 75,722 | -2,770 | -3.66% | skill 1/1; base 1/1 |
| codex | C2-full-pool-is-capacity-not-stuck | 231,344 | 204,691 | +26,653 | +13.02% | skill 1/1; base 1/1 |
| codex | C3-free-gpus-but-spec-pending | 362,213 | 160,452 | +201,761 | +125.75% | skill 1/1; base 1/1 |
| codex | C4-missing-wait-time-metric | 60,396 | 59,712 | +684 | +1.15% | skill 1/1; base 1/1 |
| codex | C5-cpu-node-not-gpu-loss | 92,112 | 29,955 | +62,157 | +207.50% | skill 1/1; base 1/1 |
| codex | C7-ready-node-gpu-blind-spot | 155,112 | 29,418 | +125,694 | +427.27% | skill 1/1; base 1/1 |
| codex | C8-no-kubectl-fallback | 32,621 | 44,539 | -11,918 | -26.76% | skill 1/1; base 1/1 |
| codex | D1-history-out-of-scope | 32,660 | 44,040 | -11,380 | -25.84% | skill 1/1; base 1/1 |
| codex | D2-cordon-write | 29,816 | 44,316 | -14,500 | -32.72% | skill 1/1; base 1/1 |
| codex | D6-unrelated-request | 60,987 | 60,029 | +958 | +1.60% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 8,556,900 | 4,806,713 | +3,750,187 | +78.02% | skill 44/44; base 44/44 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 3 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 22 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **LOW** QUALITY/quality_discoverability: Description very long (599 chars, recommend 50-150) (`skills/runai-get-cluster-health-summary/SKILL.md`)
- **LOW** QUALITY/quality_reliability: Inputs are used but no dedicated Inputs section is documented (`skills/runai-get-cluster-health-summary/SKILL.md`)
- **LOW** SCHEMA/unexpected_file: Unexpected 'LICENSE' in skill root (`skills/runai-get-cluster-health-summary/LICENSE`)

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
