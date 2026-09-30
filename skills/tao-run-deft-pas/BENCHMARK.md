# Skill Benchmark: tao-run-deft-pas

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `tao-run-deft-pas`
- Evaluation date: 2026-09-22
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 14 evaluation tasks (14 positive)
- Dataset digest: `sha256:40d7b333cefc143c30dd17c4ffbefe571c121a4003d1bf967029bf231d1321d9` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 85.7% — baseline ran, but no comparable score was available; uplift unavailable | 62.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 21.3% → 96.0% (+74.7 points) | 33.3% → 67.8% (+34.5 points) |
| Discoverability | 54.7% — baseline ran, but no comparable score was available; uplift unavailable | 0.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 25.1% → 80.2% (+55.1 points) | 25.0% → 45.4% (+20.4 points) |
| Efficiency | 97.8% — baseline ran, but no comparable score was available; uplift unavailable | 99.6% → 99.2% (-0.4 points) |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 870,435 | 924,669 | N/A | N/A | skill 15/15; base 30/30 |
| claude-code | tao-run-deft-pas-attached-loop-liveness | 31,568 | 64,088 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | tao-run-deft-pas-basic | 70,820 | 89,772 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-run-deft-pas-bounded-nested-discovery | 71,536 | 33,777 | +37,759 | +111.79% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-clean-intake | 101,365 | 94,592 | N/A | N/A | skill 2/2; base 3/3 |
| claude-code | tao-run-deft-pas-discovery-provenance | 123,679 | 97,890 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-run-deft-pas-explicit-root-ambiguity | 71,138 | 31,453 | +39,685 | +126.17% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-history-recovery-plan | 69,565 | 32,747 | +36,818 | +112.43% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-metric-contract | 71,045 | 35,159 | +35,886 | +102.07% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-plain-language-routing | 29,319 | 86,997 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-run-deft-pas-plain-language-routing-kpi-paraphrase | 29,350 | 87,090 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-run-deft-pas-plain-language-routing-no-model-name | 29,312 | 86,976 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-run-deft-pas-plain-language-routing-stage-paraphrase | 29,347 | 87,081 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-run-deft-pas-resume-plan | 69,971 | 30,946 | +39,025 | +126.11% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-virtualenv-contract | 72,420 | 66,101 | N/A | N/A | skill 1/1; base 2/2 |
| codex | All cases | 327,187 | 451,025 | N/A | N/A | skill 23/23; base 33/33 |
| codex | tao-run-deft-pas-attached-loop-liveness | 13,906 | 13,518 | +388 | +2.87% | skill 1/1; base 1/1 |
| codex | tao-run-deft-pas-basic | 13,855 | 40,752 | N/A | N/A | skill 1/1; base 3/3 |
| codex | tao-run-deft-pas-bounded-nested-discovery | 43,325 | 28,212 | N/A | N/A | skill 3/3; base 2/2 |
| codex | tao-run-deft-pas-clean-intake | 42,332 | 41,026 | +1,306 | +3.18% | skill 3/3; base 3/3 |
| codex | tao-run-deft-pas-discovery-provenance | 28,846 | 41,977 | N/A | N/A | skill 2/2; base 3/3 |
| codex | tao-run-deft-pas-explicit-root-ambiguity | 14,109 | 13,584 | +525 | +3.86% | skill 1/1; base 1/1 |
| codex | tao-run-deft-pas-history-recovery-plan | 14,226 | 13,998 | +228 | +1.63% | skill 1/1; base 1/1 |
| codex | tao-run-deft-pas-metric-contract | 15,569 | 14,720 | +849 | +5.77% | skill 1/1; base 1/1 |
| codex | tao-run-deft-pas-plain-language-routing | 13,532 | 39,666 | N/A | N/A | skill 1/1; base 3/3 |
| codex | tao-run-deft-pas-plain-language-routing-kpi-paraphrase | 13,489 | 39,890 | N/A | N/A | skill 1/1; base 3/3 |
| codex | tao-run-deft-pas-plain-language-routing-no-model-name | 13,463 | 39,720 | N/A | N/A | skill 1/1; base 3/3 |
| codex | tao-run-deft-pas-plain-language-routing-stage-paraphrase | 13,481 | 39,921 | N/A | N/A | skill 1/1; base 3/3 |
| codex | tao-run-deft-pas-resume-plan | 43,233 | 41,577 | +1,656 | +3.98% | skill 3/3; base 3/3 |
| codex | tao-run-deft-pas-virtualenv-contract | 43,821 | 42,464 | +1,357 | +3.20% | skill 3/3; base 3/3 |
| ALL AGENTS | Dataset aggregate | 1,197,622 | 1,375,694 | N/A | N/A | skill 38/38; base 63/63 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 122 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 4 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 14 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/audit_deft_run.py and scripts/commit_stage.py:
  "_results_root_for_scope()" in scripts/audit_deft_run.py (lines 607-613)
  vs "_results_root_for_scope()" in scripts/commit_stage.py (lines 347-353) (`scripts/audit_deft_run.py:607`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/init_deft_state.py and scripts/prepare_deft_config.py and scripts/run_deft_container.py:
  "_workspace_child()" in scripts/init_deft_state.py (lines 123-131)
  vs "_workspace_child()" in scripts/prepare_deft_config.py (lines 74-82)
  vs "_workspace_child()" in scripts/run_deft_container.py (lines 122-130) (`scripts/init_deft_state.py:123`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/deft_action_contract.py and scripts/run_deft_container.py:
  "fresh_output_path()" in scripts/deft_action_contract.py (lines 125-141)
  vs "_fresh_output_path()" in scripts/run_deft_container.py (lines 82-103) (`scripts/deft_action_contract.py:125`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/deft_action_contract.py and scripts/run_deft_container.py:
  "launch_label()" in scripts/deft_action_contract.py (lines 263-275)
  vs "_launch_label()" in scripts/run_deft_container.py (lines 407-419) (`scripts/deft_action_contract.py:263`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/applications/tao-run-deft-pas/SKILL.md`)
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
