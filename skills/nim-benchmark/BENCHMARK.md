# Skill Benchmark: nim-benchmark

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nim-benchmark`
- Validation status: `passed`
- Evaluation date: 2026-11-25
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 24 evaluation tasks (24 positive)
- Dataset digest: `sha256:fba927bb52e9e44c2bacb40398ca881bc41ac987d716803379a3b338ced8953f` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nim-benchmark | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nim-benchmark | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nim-benchmark | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nim-benchmark | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nim-benchmark | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `nim-benchmark`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 82.9% — baseline ran, but no comparable score was available; uplift unavailable | 83.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 83.3% → 95.8% (+12.5 points) | 72.9% → 93.8% (+20.9 points) |
| Correctness | 35.8% → 88.3% (+52.5 points) | 28.3% → 82.5% (+54.2 points) |
| Discoverability | 80.0% — baseline ran, but no comparable score was available; uplift unavailable | 86.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 30.4% → 64.1% (+33.7 points) | 25.3% → 59.9% (+34.6 points) |
| Efficiency | 86.3% — baseline ran, but no comparable score was available; uplift unavailable | 93.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.9% → 96.2% (-1.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 90.5% → 91.0% (+0.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,897,094 | 3,339,939 | -442,845 | -13.26% | skill 24/24; base 24/24 |
| claude-code | neg-deploy-the-nim | 145,524 | 144,512 | +1,012 | +0.70% | skill 1/1; base 1/1 |
| claude-code | neg-pick-a-nim | 72,290 | 36,304 | +35,986 | +99.12% | skill 1/1; base 1/1 |
| claude-code | neg-tune-the-nim | 75,205 | 144,594 | -69,389 | -47.99% | skill 1/1; base 1/1 |
| claude-code | real-accuracy-after-tune | 204,167 | 180,899 | +23,268 | +12.86% | skill 1/1; base 1/1 |
| claude-code | real-accuracy-cleanup-failure-with-other-sets | 160,395 | 146,056 | +14,339 | +9.82% | skill 1/1; base 1/1 |
| claude-code | real-baseline-auth-options-stay-in-approved-plan | 115,497 | 182,397 | -66,900 | -36.68% | skill 1/1; base 1/1 |
| claude-code | real-both-no-baseline | 114,642 | 144,840 | -30,198 | -20.85% | skill 1/1; base 1/1 |
| claude-code | real-folder-baseline-needs-no-endpoint-credentials | 74,630 | 107,284 | -32,654 | -30.44% | skill 1/1; base 1/1 |
| claude-code | real-invalid-baseline-evidence-clears-old-verdict | 157,583 | 143,120 | +14,463 | +10.11% | skill 1/1; base 1/1 |
| claude-code | real-pod-export-read-access-failure | 116,824 | 107,530 | +9,294 | +8.64% | skill 1/1; base 1/1 |
| claude-code | real-speed-with-baseline-endpoint | 155,286 | 144,152 | +11,134 | +7.72% | skill 1/1; base 1/1 |
| claude-code | real-startup-failure-with-no-log | 116,037 | 107,709 | +8,328 | +7.73% | skill 1/1; base 1/1 |
| claude-code | real-timeout-with-no-console-log | 115,615 | 107,306 | +8,309 | +7.74% | skill 1/1; base 1/1 |
| claude-code | real-vlm-accuracy | 155,598 | 219,500 | -63,902 | -29.11% | skill 1/1; base 1/1 |
| claude-code | real-vlm-speed | 118,325 | 349,190 | -230,865 | -66.11% | skill 1/1; base 1/1 |
| claude-code | trick-accuracy-client-not-stopped | 116,147 | 107,070 | +9,077 | +8.48% | skill 1/1; base 1/1 |
| claude-code | trick-cleanup-failure-with-valid-export | 115,520 | 107,384 | +8,136 | +7.58% | skill 1/1; base 1/1 |
| claude-code | trick-export-tool-version-mismatch | 115,007 | 143,245 | -28,238 | -19.71% | skill 1/1; base 1/1 |
| claude-code | trick-key-on-command-line | 75,072 | 106,975 | -31,903 | -29.82% | skill 1/1; base 1/1 |
| claude-code | trick-mismatched-baseline | 113,724 | 107,198 | +6,526 | +6.09% | skill 1/1; base 1/1 |
| claude-code | trick-old-aiperf-before-approval | 156,518 | 182,047 | -25,529 | -14.02% | skill 1/1; base 1/1 |
| claude-code | trick-run-aiperf-directly | 75,582 | 106,496 | -30,914 | -29.03% | skill 1/1; base 1/1 |
| claude-code | trick-self-judging | 114,742 | 107,058 | +7,684 | +7.18% | skill 1/1; base 1/1 |
| claude-code | trick-timeout-stop-and-cleanup-both-fail | 117,164 | 107,073 | +10,091 | +9.42% | skill 1/1; base 1/1 |
| codex | All cases | 1,532,083 | 1,811,594 | -279,511 | -15.43% | skill 24/24; base 24/24 |
| codex | neg-deploy-the-nim | 78,710 | 77,359 | +1,351 | +1.75% | skill 1/1; base 1/1 |
| codex | neg-pick-a-nim | 38,618 | 47,309 | -8,691 | -18.37% | skill 1/1; base 1/1 |
| codex | neg-tune-the-nim | 48,133 | 215,029 | -166,896 | -77.62% | skill 1/1; base 1/1 |
| codex | real-accuracy-after-tune | 90,541 | 80,654 | +9,887 | +12.26% | skill 1/1; base 1/1 |
| codex | real-accuracy-cleanup-failure-with-other-sets | 85,918 | 60,105 | +25,813 | +42.95% | skill 1/1; base 1/1 |
| codex | real-baseline-auth-options-stay-in-approved-plan | 29,507 | 56,160 | -26,653 | -47.46% | skill 1/1; base 1/1 |
| codex | real-both-no-baseline | 68,635 | 45,347 | +23,288 | +51.36% | skill 1/1; base 1/1 |
| codex | real-folder-baseline-needs-no-endpoint-credentials | 29,343 | 54,650 | -25,307 | -46.31% | skill 1/1; base 1/1 |
| codex | real-invalid-baseline-evidence-clears-old-verdict | 182,153 | 53,812 | +128,341 | +238.50% | skill 1/1; base 1/1 |
| codex | real-pod-export-read-access-failure | 87,000 | 57,094 | +29,906 | +52.38% | skill 1/1; base 1/1 |
| codex | real-speed-with-baseline-endpoint | 67,664 | 88,785 | -21,121 | -23.79% | skill 1/1; base 1/1 |
| codex | real-startup-failure-with-no-log | 47,756 | 13,435 | +34,321 | +255.46% | skill 1/1; base 1/1 |
| codex | real-timeout-with-no-console-log | 124,640 | 67,178 | +57,462 | +85.54% | skill 1/1; base 1/1 |
| codex | real-vlm-accuracy | 47,639 | 210,424 | -162,785 | -77.36% | skill 1/1; base 1/1 |
| codex | real-vlm-speed | 85,594 | 272,244 | -186,650 | -68.56% | skill 1/1; base 1/1 |
| codex | trick-accuracy-client-not-stopped | 47,377 | 58,330 | -10,953 | -18.78% | skill 1/1; base 1/1 |
| codex | trick-cleanup-failure-with-valid-export | 62,683 | 13,430 | +49,253 | +366.74% | skill 1/1; base 1/1 |
| codex | trick-export-tool-version-mismatch | 47,804 | 44,819 | +2,985 | +6.66% | skill 1/1; base 1/1 |
| codex | trick-key-on-command-line | 29,466 | 40,709 | -11,243 | -27.62% | skill 1/1; base 1/1 |
| codex | trick-mismatched-baseline | 62,469 | 54,904 | +7,565 | +13.78% | skill 1/1; base 1/1 |
| codex | trick-old-aiperf-before-approval | 63,413 | 76,025 | -12,612 | -16.59% | skill 1/1; base 1/1 |
| codex | trick-run-aiperf-directly | 29,510 | 26,785 | +2,725 | +10.17% | skill 1/1; base 1/1 |
| codex | trick-self-judging | 29,608 | 54,568 | -24,960 | -45.74% | skill 1/1; base 1/1 |
| codex | trick-timeout-stop-and-cleanup-both-fail | 47,902 | 42,439 | +5,463 | +12.87% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 4,429,177 | 5,151,533 | -722,356 | -14.02% | skill 48/48; base 48/48 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 63 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 24 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** DUPLICATE/duplicate: Duplicate content found across tests/test_nb_accuracy.py and tests/test_nb_perf.py:
  "test_a_timeout_removes_the_container()" in tests/test_nb_accuracy.py (lines 257-269)
  vs "test_a_timeout_removes_the_container()" in tests/test_nb_perf.py (lines 263-275) (`tests/test_nb_accuracy.py:257`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/nim-benchmark/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/nim-benchmark/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nim-benchmark/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nim-benchmark/SKILL.md`)
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
