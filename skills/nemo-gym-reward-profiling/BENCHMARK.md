# Skill Benchmark: nemo-gym-reward-profiling

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemo-gym-reward-profiling`
- Validation status: `passed`
- Evaluation date: 2026-11-20
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 17 evaluation tasks (13 positive, 4 negative)
- Dataset digest: `sha256:4786da0fd072041d60b9f858457929a19d08378369a7c9c8928b967b0942b9fc` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nemo-gym-reward-profiling | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nemo-gym-reward-profiling | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nemo-gym-reward-profiling | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nemo-gym-reward-profiling | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nemo-gym-reward-profiling | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `nemo-gym-reward-profiling`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | Not available | 90.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 100.0% → 100.0% (±0.0 points) |
| Correctness | Not available | 82.4% → 97.7% (+15.3 points) |
| Discoverability | Not available | 81.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 60.7% → 90.4% (+29.7 points) |
| Efficiency | Not available | 84.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 86.9% → 87.1% (+0.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,037,752 | 1,868,812 | +168,940 | +9.04% | skill 17/17; base 17/17 |
| claude-code | check-new-environment-before-rl | 118,989 | 108,357 | +10,632 | +9.81% | skill 1/1; base 1/1 |
| claude-code | choose-training-tasks-partial-run | 255,096 | 160,230 | +94,866 | +59.21% | skill 1/1; base 1/1 |
| claude-code | headline-mean-from-agent-metrics | 121,736 | 109,928 | +11,808 | +10.74% | skill 1/1; base 1/1 |
| claude-code | masked-rollouts-choose-tasks | 208,014 | 146,799 | +61,215 | +41.70% | skill 1/1; base 1/1 |
| claude-code | negative-crashed-run | 36,591 | 36,122 | +469 | +1.30% | skill 1/1; base 1/1 |
| claude-code | negative-repeats-to-compare-checkpoints | 36,639 | 36,744 | -105 | -0.29% | skill 1/1; base 1/1 |
| claude-code | negative-score-gap-vs-model-card | 36,030 | 35,797 | +233 | +0.65% | skill 1/1; base 1/1 |
| claude-code | negative-server-latency-profiling | 107,970 | 71,925 | +36,045 | +50.11% | skill 1/1; base 1/1 |
| claude-code | never-pass-share-from-profile-only | 162,921 | 146,166 | +16,755 | +11.46% | skill 1/1; base 1/1 |
| claude-code | pass-rate-of-partial-run | 201,302 | 201,000 | +302 | +0.15% | skill 1/1; base 1/1 |
| claude-code | profile-command-and-outputs | 77,595 | 143,877 | -66,282 | -46.07% | skill 1/1; base 1/1 |
| claude-code | profile-missing-rollout-keys-error | 77,055 | 142,946 | -65,891 | -46.10% | skill 1/1; base 1/1 |
| claude-code | repeat-coverage-from-repeat-metrics | 127,856 | 111,100 | +16,756 | +15.08% | skill 1/1; base 1/1 |
| claude-code | repeat-missing-count-blind-spot | 111,748 | 113,626 | -1,878 | -1.65% | skill 1/1; base 1/1 |
| claude-code | same-mean-different-spread | 161,513 | 117,383 | +44,130 | +37.59% | skill 1/1; base 1/1 |
| claude-code | single-rollout-zero-std | 119,378 | 150,753 | -31,375 | -20.81% | skill 1/1; base 1/1 |
| claude-code | too-few-repeats-for-hard-tasks | 77,319 | 36,059 | +41,260 | +114.42% | skill 1/1; base 1/1 |
| codex | All cases | 1,385,914 | 1,228,694 | +157,220 | +12.80% | skill 17/17; base 17/17 |
| codex | check-new-environment-before-rl | 83,439 | 51,507 | +31,932 | +62.00% | skill 1/1; base 1/1 |
| codex | choose-training-tasks-partial-run | 114,899 | 109,151 | +5,748 | +5.27% | skill 1/1; base 1/1 |
| codex | headline-mean-from-agent-metrics | 71,984 | 43,583 | +28,401 | +65.17% | skill 1/1; base 1/1 |
| codex | masked-rollouts-choose-tasks | 118,145 | 211,632 | -93,487 | -44.17% | skill 1/1; base 1/1 |
| codex | negative-crashed-run | 57,171 | 30,491 | +26,680 | +87.50% | skill 1/1; base 1/1 |
| codex | negative-repeats-to-compare-checkpoints | 156,734 | 38,924 | +117,810 | +302.67% | skill 1/1; base 1/1 |
| codex | negative-score-gap-vs-model-card | 61,733 | 60,222 | +1,511 | +2.51% | skill 1/1; base 1/1 |
| codex | negative-server-latency-profiling | 74,381 | 77,533 | -3,152 | -4.07% | skill 1/1; base 1/1 |
| codex | never-pass-share-from-profile-only | 69,599 | 60,802 | +8,797 | +14.47% | skill 1/1; base 1/1 |
| codex | pass-rate-of-partial-run | 87,023 | 106,640 | -19,617 | -18.40% | skill 1/1; base 1/1 |
| codex | profile-command-and-outputs | 62,766 | 52,417 | +10,349 | +19.74% | skill 1/1; base 1/1 |
| codex | profile-missing-rollout-keys-error | 66,862 | 65,567 | +1,295 | +1.98% | skill 1/1; base 1/1 |
| codex | repeat-coverage-from-repeat-metrics | 82,170 | 42,073 | +40,097 | +95.30% | skill 1/1; base 1/1 |
| codex | repeat-missing-count-blind-spot | 80,738 | 100,985 | -20,247 | -20.05% | skill 1/1; base 1/1 |
| codex | same-mean-different-spread | 71,554 | 67,691 | +3,863 | +5.71% | skill 1/1; base 1/1 |
| codex | single-rollout-zero-std | 70,074 | 78,780 | -8,706 | -11.05% | skill 1/1; base 1/1 |
| codex | too-few-repeats-for-hard-tasks | 56,642 | 30,696 | +25,946 | +84.53% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 3,423,666 | 3,097,506 | +326,160 | +10.53% | skill 34/34; base 34/34 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 8 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 17 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/nemo-gym-reward-profiling/SKILL.md`)
- **MEDIUM** SECURITY/Skill declares no tool scope ('permissions' or 'allowed-tools') but code capabilities were detected: file_read. (LP3): MCP Least Privilege: Skill declares no tool scope ('permissions' or 'allowed-tools') but code capabilities were detected: file_read. (`SKILL.md:1`)
- **LOW** QUALITY/quality_discoverability: Description very long (420 chars, recommend 50-150) (`skills/nemo-gym-reward-profiling/SKILL.md`)
- **LOW** QUALITY/quality_reliability: Inputs are used but no dedicated Inputs section is documented (`skills/nemo-gym-reward-profiling/SKILL.md`)
- **LOW** QUALITY/quality_reliability: Structured output is used but no dedicated Output Format section is documented (`skills/nemo-gym-reward-profiling/SKILL.md`)
- 3 additional finding(s) are available in the full evaluation artifacts.

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
