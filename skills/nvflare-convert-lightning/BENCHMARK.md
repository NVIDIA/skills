# Skill Benchmark: nvflare-convert-lightning

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvflare-convert-lightning`
- Evaluation date: 2026-10-06
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 22 evaluation tasks (22 positive)
- Dataset digest: `sha256:27fefd3863135e758ee738361deca2313b8f1efa03a456440267ac92dc654196` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 77.0% — baseline ran, but no comparable score was available; uplift unavailable | 64.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 54.6% → 70.5% (+15.9 points) | 18.2% → 36.4% (+18.2 points) |
| Correctness | 86.4% → 89.1% (+2.7 points) | 77.3% → 81.8% (+4.5 points) |
| Discoverability | 73.2% — baseline ran, but no comparable score was available; uplift unavailable | 60.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 72.7% → 78.4% (+5.7 points) | 61.2% → 74.8% (+13.6 points) |
| Efficiency | 73.9% — baseline ran, but no comparable score was available; uplift unavailable | 69.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 33,654,559 | 60,047,323 | -26,392,764 | -43.95% | skill 22/22; base 22/22 |
| claude-code | lightning-combined-fedstats-workflow-boundary | 100,155 | 216,887 | -116,732 | -53.82% | skill 1/1; base 1/1 |
| claude-code | lightning-convert-basic | 1,374,821 | 2,434,829 | -1,060,008 | -43.54% | skill 1/1; base 1/1 |
| claude-code | lightning-convert-custom-aggregation | 2,330,415 | 3,535,639 | -1,205,224 | -34.09% | skill 1/1; base 1/1 |
| claude-code | lightning-convert-external-data-path | 2,824,386 | 2,818,102 | +6,284 | +0.22% | skill 1/1; base 1/1 |
| claude-code | lightning-convert-with-eval | 1,753,172 | 3,262,692 | -1,509,520 | -46.27% | skill 1/1; base 1/1 |
| claude-code | lightning-custom-aggregation-with-server-metrics | 7,165,122 | 9,197,573 | -2,032,451 | -22.10% | skill 1/1; base 1/1 |
| claude-code | lightning-data-derived-required-arg | 2,434,896 | 6,885,340 | -4,450,444 | -64.64% | skill 1/1; base 1/1 |
| claude-code | lightning-ddp-multigpu | 1,422,134 | 2,214,464 | -792,330 | -35.78% | skill 1/1; base 1/1 |
| claude-code | lightning-device-selection | 2,232,036 | 6,400,482 | -4,168,446 | -65.13% | skill 1/1; base 1/1 |
| claude-code | lightning-eval-only | 2,596,354 | 1,964,639 | +631,715 | +32.15% | skill 1/1; base 1/1 |
| claude-code | lightning-factory-owner-routing | 1,393,404 | 3,465,923 | -2,072,519 | -59.80% | skill 1/1; base 1/1 |
| claude-code | lightning-global-negative-kubernetes | 803,589 | 3,720,589 | -2,917,000 | -78.40% | skill 1/1; base 1/1 |
| claude-code | lightning-loss-only-selection-metric | 2,889,563 | 3,613,602 | -724,039 | -20.04% | skill 1/1; base 1/1 |
| claude-code | lightning-negative-ddp-without-federation | 249,848 | 248,966 | +882 | +0.35% | skill 1/1; base 1/1 |
| claude-code | lightning-negative-dual-trainer | 64,109 | 1,957,157 | -1,893,048 | -96.72% | skill 1/1; base 1/1 |
| claude-code | lightning-negative-huggingface | 120,855 | 738,102 | -617,247 | -83.63% | skill 1/1; base 1/1 |
| claude-code | lightning-negative-inference-serving | 736,745 | 1,132,762 | -396,017 | -34.96% | skill 1/1; base 1/1 |
| claude-code | lightning-negative-plain-pytorch | 154,345 | 246,018 | -91,673 | -37.26% | skill 1/1; base 1/1 |
| claude-code | lightning-negative-profiling | 973,866 | 1,486,617 | -512,751 | -34.49% | skill 1/1; base 1/1 |
| claude-code | lightning-negative-tensorflow-keras | 218,100 | 286,974 | -68,874 | -24.00% | skill 1/1; base 1/1 |
| claude-code | lightning-negative-training-loop-change | 191,286 | 186,683 | +4,603 | +2.47% | skill 1/1; base 1/1 |
| claude-code | lightning-positive-implicit-federation-intent | 1,625,358 | 4,033,283 | -2,407,925 | -59.70% | skill 1/1; base 1/1 |
| codex | All cases | 23,217,747 | 31,987,239 | -8,769,492 | -27.42% | skill 22/22; base 22/22 |
| codex | lightning-combined-fedstats-workflow-boundary | 30,113 | 1,845,805 | -1,815,692 | -98.37% | skill 1/1; base 1/1 |
| codex | lightning-convert-basic | 1,526,572 | 4,222,071 | -2,695,499 | -63.84% | skill 1/1; base 1/1 |
| codex | lightning-convert-custom-aggregation | 2,400,791 | 2,385,463 | +15,328 | +0.64% | skill 1/1; base 1/1 |
| codex | lightning-convert-external-data-path | 1,306,246 | 2,364,934 | -1,058,688 | -44.77% | skill 1/1; base 1/1 |
| codex | lightning-convert-with-eval | 1,279,164 | 3,004,821 | -1,725,657 | -57.43% | skill 1/1; base 1/1 |
| codex | lightning-custom-aggregation-with-server-metrics | 2,614,557 | 3,664,527 | -1,049,970 | -28.65% | skill 1/1; base 1/1 |
| codex | lightning-data-derived-required-arg | 1,180,276 | 1,599,244 | -418,968 | -26.20% | skill 1/1; base 1/1 |
| codex | lightning-ddp-multigpu | 2,104,884 | 772,529 | +1,332,355 | +172.47% | skill 1/1; base 1/1 |
| codex | lightning-device-selection | 792,540 | 1,939,823 | -1,147,283 | -59.14% | skill 1/1; base 1/1 |
| codex | lightning-eval-only | 3,045,654 | 290,261 | +2,755,393 | +949.28% | skill 1/1; base 1/1 |
| codex | lightning-factory-owner-routing | 1,460,589 | 1,307,645 | +152,944 | +11.70% | skill 1/1; base 1/1 |
| codex | lightning-global-negative-kubernetes | 349,480 | 194,085 | +155,395 | +80.07% | skill 1/1; base 1/1 |
| codex | lightning-loss-only-selection-metric | 1,119,534 | 1,623,782 | -504,248 | -31.05% | skill 1/1; base 1/1 |
| codex | lightning-negative-ddp-without-federation | 239,097 | 206,274 | +32,823 | +15.91% | skill 1/1; base 1/1 |
| codex | lightning-negative-dual-trainer | 139,187 | 979,506 | -840,319 | -85.79% | skill 1/1; base 1/1 |
| codex | lightning-negative-huggingface | 97,419 | 1,682,866 | -1,585,447 | -94.21% | skill 1/1; base 1/1 |
| codex | lightning-negative-inference-serving | 274,395 | 471,111 | -196,716 | -41.76% | skill 1/1; base 1/1 |
| codex | lightning-negative-plain-pytorch | 101,870 | 410,569 | -308,699 | -75.19% | skill 1/1; base 1/1 |
| codex | lightning-negative-profiling | 1,090,706 | 990,991 | +99,715 | +10.06% | skill 1/1; base 1/1 |
| codex | lightning-negative-tensorflow-keras | 107,786 | 1,581,940 | -1,474,154 | -93.19% | skill 1/1; base 1/1 |
| codex | lightning-negative-training-loop-change | 177,433 | 119,162 | +58,271 | +48.90% | skill 1/1; base 1/1 |
| codex | lightning-positive-implicit-federation-intent | 1,779,454 | 329,830 | +1,449,624 | +439.51% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 56,872,306 | 92,034,562 | -35,162,256 | -38.21% | skill 44/44; base 44/44 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 8 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 22 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nvflare-convert-lightning/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nvflare-convert-lightning/SKILL.md`)
- **LOW** QUALITY/quality_correctness: No examples provided (`skills/nvflare-convert-lightning/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (632 chars, recommend 50-150) (`skills/nvflare-convert-lightning/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/nvflare-convert-lightning/SKILL.md`)
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
