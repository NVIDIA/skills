# Skill Benchmark: nvflare-convert-pytorch

> **Overall verdict: NEUTRAL — One or more dimensions remain below PASS**

Live evaluation did not show a material gain or regression. Collect more evidence or improve the skill before making a publication decision.

## Evaluation Metadata

- Skill: `nvflare-convert-pytorch`
- Evaluation date: 2026-10-06
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 17 evaluation tasks (17 positive)
- Dataset digest: `sha256:d42be143c4af4e5485ddeff65048069352998e11e2aa2087be4ab06c32675c38` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 77.5% — baseline ran, but no comparable score was available; uplift unavailable | 64.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 58.8% → 47.1% (-11.7 points) | 29.4% → 23.5% (-5.9 points) |
| Correctness | 91.8% → 95.3% (+3.5 points) | 81.2% → 90.6% (+9.4 points) |
| Discoverability | 89.9% — baseline ran, but no comparable score was available; uplift unavailable | 71.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 75.3% → 82.3% (+7.0 points) | 70.6% → 66.5% (-4.1 points) |
| Efficiency | 73.2% — baseline ran, but no comparable score was available; uplift unavailable | 71.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 27,073,573 | 42,068,925 | -14,995,352 | -35.64% | skill 17/17; base 17/17 |
| claude-code | pytorch-combined-fedstats-workflow-boundary | 97,260 | 124,403 | -27,143 | -21.82% | skill 1/1; base 1/1 |
| claude-code | pytorch-convert-basic | 1,959,727 | 1,602,324 | +357,403 | +22.31% | skill 1/1; base 1/1 |
| claude-code | pytorch-convert-custom-aggregation | 4,632,131 | 7,409,017 | -2,776,886 | -37.48% | skill 1/1; base 1/1 |
| claude-code | pytorch-convert-external-data-path | 1,915,330 | 1,266,163 | +649,167 | +51.27% | skill 1/1; base 1/1 |
| claude-code | pytorch-convert-with-eval | 1,015,218 | 697,840 | +317,378 | +45.48% | skill 1/1; base 1/1 |
| claude-code | pytorch-dataparallel-in-process | 2,240,355 | 5,310,115 | -3,069,760 | -57.81% | skill 1/1; base 1/1 |
| claude-code | pytorch-device-selection | 1,486,340 | 5,068,205 | -3,581,865 | -70.67% | skill 1/1; base 1/1 |
| claude-code | pytorch-global-negative-kubernetes | 2,828,263 | 990,809 | +1,837,454 | +185.45% | skill 1/1; base 1/1 |
| claude-code | pytorch-injection-resistance | 1,017,884 | 881,314 | +136,570 | +15.50% | skill 1/1; base 1/1 |
| claude-code | pytorch-iterative-rerun | 2,626,271 | 4,494,068 | -1,867,797 | -41.56% | skill 1/1; base 1/1 |
| claude-code | pytorch-missing-evaluation-fail-closed | 1,926,075 | 366,050 | +1,560,025 | +426.18% | skill 1/1; base 1/1 |
| claude-code | pytorch-negative-huggingface | 226,290 | 246,037 | -19,747 | -8.03% | skill 1/1; base 1/1 |
| claude-code | pytorch-negative-lightning | 230,646 | 150,527 | +80,119 | +53.23% | skill 1/1; base 1/1 |
| claude-code | pytorch-optimizer-factory-owner-routing | 994,950 | 2,541,952 | -1,547,002 | -60.86% | skill 1/1; base 1/1 |
| claude-code | pytorch-safe-checkpoint-load | 1,569,009 | 4,138,026 | -2,569,017 | -62.08% | skill 1/1; base 1/1 |
| claude-code | pytorch-state-dict-schema-mismatch | 1,043,977 | 471,106 | +572,871 | +121.60% | skill 1/1; base 1/1 |
| claude-code | pytorch-unsupported-privacy-request | 1,263,847 | 6,310,969 | -5,047,122 | -79.97% | skill 1/1; base 1/1 |
| codex | All cases | 14,990,261 | 18,365,006 | -3,374,745 | -18.38% | skill 17/17; base 17/17 |
| codex | pytorch-combined-fedstats-workflow-boundary | 46,522 | 89,797 | -43,275 | -48.19% | skill 1/1; base 1/1 |
| codex | pytorch-convert-basic | 1,450,430 | 1,742,852 | -292,422 | -16.78% | skill 1/1; base 1/1 |
| codex | pytorch-convert-custom-aggregation | 1,981,346 | 2,842,640 | -861,294 | -30.30% | skill 1/1; base 1/1 |
| codex | pytorch-convert-external-data-path | 1,481,788 | 1,452,550 | +29,238 | +2.01% | skill 1/1; base 1/1 |
| codex | pytorch-convert-with-eval | 974,838 | 911,574 | +63,264 | +6.94% | skill 1/1; base 1/1 |
| codex | pytorch-dataparallel-in-process | 940,281 | 1,856,085 | -915,804 | -49.34% | skill 1/1; base 1/1 |
| codex | pytorch-device-selection | 790,127 | 1,178,147 | -388,020 | -32.93% | skill 1/1; base 1/1 |
| codex | pytorch-global-negative-kubernetes | 366,795 | 343,222 | +23,573 | +6.87% | skill 1/1; base 1/1 |
| codex | pytorch-injection-resistance | 631,719 | 2,179,727 | -1,548,008 | -71.02% | skill 1/1; base 1/1 |
| codex | pytorch-iterative-rerun | 1,497,275 | 286,996 | +1,210,279 | +421.71% | skill 1/1; base 1/1 |
| codex | pytorch-missing-evaluation-fail-closed | 595,589 | 218,591 | +376,998 | +172.47% | skill 1/1; base 1/1 |
| codex | pytorch-negative-huggingface | 425,029 | 1,401,989 | -976,960 | -69.68% | skill 1/1; base 1/1 |
| codex | pytorch-negative-lightning | 71,100 | 68,376 | +2,724 | +3.98% | skill 1/1; base 1/1 |
| codex | pytorch-optimizer-factory-owner-routing | 1,120,534 | 1,831,805 | -711,271 | -38.83% | skill 1/1; base 1/1 |
| codex | pytorch-safe-checkpoint-load | 1,474,719 | 1,494,388 | -19,669 | -1.32% | skill 1/1; base 1/1 |
| codex | pytorch-state-dict-schema-mismatch | 430,830 | 213,163 | +217,667 | +102.11% | skill 1/1; base 1/1 |
| codex | pytorch-unsupported-privacy-request | 711,339 | 253,104 | +458,235 | +181.05% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 42,063,834 | 60,433,931 | -18,370,097 | -30.40% | skill 34/34; base 34/34 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 8 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **NEUTRAL** | 2 agent(s); 17 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nvflare-convert-pytorch/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nvflare-convert-pytorch/SKILL.md`)
- **LOW** QUALITY/quality_correctness: No examples provided (`skills/nvflare-convert-pytorch/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (362 chars, recommend 50-150) (`skills/nvflare-convert-pytorch/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/nvflare-convert-pytorch/SKILL.md`)
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
