# Skill Benchmark: tao-run-deft-pas

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `tao-run-deft-pas`
- Evaluation date: 2026-10-16
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 20 evaluation tasks (20 positive)
- Dataset digest: `sha256:a96fe279f8dcd9995cf48dab59613268866481e77e344ffcace5746b56926055` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 89.4% — baseline ran, but no comparable score was available; uplift unavailable | 69.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 35.0% → 95.0% (+60.0 points) | 45.0% → 79.0% (+34.0 points) |
| Discoverability | 67.0% — baseline ran, but no comparable score was available; uplift unavailable | 9.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 30.0% → 85.5% (+55.5 points) | 29.4% → 59.8% (+30.4 points) |
| Efficiency | 99.5% — baseline ran, but no comparable score was available; uplift unavailable | 100.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,168,680 | 1,497,647 | N/A | N/A | skill 17/20; base 17/20 |
| claude-code | tao-run-deft-pas-archive-search-capability-boundary | 130,374 | 282,120 | -151,746 | -53.79% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-attached-loop-liveness | 67,115 | 31,512 | +35,603 | +112.98% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-basic | 69,307 | 29,873 | +39,434 | +132.01% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-bounded-nested-discovery | 70,606 | 33,981 | +36,625 | +107.78% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-clean-intake | 70,344 | 31,310 | +39,034 | +124.67% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-completion-result-contract | 739,987 | 278,388 | +461,599 | +165.81% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-control-prerequisites-before-approval | 187,786 | 155,483 | +32,303 | +20.78% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-discovery-provenance | 72,472 | 33,257 | +39,215 | +117.92% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-explicit-root-ambiguity | 73,128 | 32,469 | +40,659 | +125.22% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-five-platform-intake | 67,680 | 154,607 | -86,927 | -56.22% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-history-recovery-plan | 33,157 | 31,957 | +1,200 | +3.76% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-image-acquisition-credential-contract | 189,941 | 156,498 | +33,443 | +21.37% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-metric-contract | 71,610 | 34,444 | +37,166 | +107.90% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-plain-language-routing | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | tao-run-deft-pas-plain-language-routing-kpi-paraphrase | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | tao-run-deft-pas-plain-language-routing-no-model-name | N/A | 116,049 | N/A | N/A | skill 0/1; base 1/1 |
| claude-code | tao-run-deft-pas-plain-language-routing-stage-paraphrase | 117,328 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| claude-code | tao-run-deft-pas-resume-plan | 69,664 | 31,652 | +38,012 | +120.09% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-untrusted-dataset-instruction | 67,341 | 30,162 | +37,179 | +123.26% | skill 1/1; base 1/1 |
| claude-code | tao-run-deft-pas-virtualenv-contract | 70,840 | 33,885 | +36,955 | +109.06% | skill 1/1; base 1/1 |
| codex | All cases | 374,045 | 276,207 | N/A | N/A | skill 17/20; base 17/20 |
| codex | tao-run-deft-pas-archive-search-capability-boundary | 14,195 | 13,994 | +201 | +1.44% | skill 1/1; base 1/1 |
| codex | tao-run-deft-pas-attached-loop-liveness | 13,844 | 13,852 | -8 | -0.06% | skill 1/1; base 1/1 |
| codex | tao-run-deft-pas-basic | 13,842 | 13,535 | +307 | +2.27% | skill 1/1; base 1/1 |
| codex | tao-run-deft-pas-bounded-nested-discovery | 14,439 | 13,771 | +668 | +4.85% | skill 1/1; base 1/1 |
| codex | tao-run-deft-pas-clean-intake | 13,863 | 13,734 | +129 | +0.94% | skill 1/1; base 1/1 |
| codex | tao-run-deft-pas-completion-result-contract | 90,360 | 13,969 | +76,391 | +546.86% | skill 1/1; base 1/1 |
| codex | tao-run-deft-pas-control-prerequisites-before-approval | 14,414 | 14,176 | +238 | +1.68% | skill 1/1; base 1/1 |
| codex | tao-run-deft-pas-discovery-provenance | 14,033 | 14,023 | +10 | +0.07% | skill 1/1; base 1/1 |
| codex | tao-run-deft-pas-explicit-root-ambiguity | 14,188 | 13,694 | +494 | +3.61% | skill 1/1; base 1/1 |
| codex | tao-run-deft-pas-five-platform-intake | 30,441 | 14,065 | +16,376 | +116.43% | skill 1/1; base 1/1 |
| codex | tao-run-deft-pas-history-recovery-plan | 14,213 | 13,960 | +253 | +1.81% | skill 1/1; base 1/1 |
| codex | tao-run-deft-pas-image-acquisition-credential-contract | 14,282 | 13,706 | +576 | +4.20% | skill 1/1; base 1/1 |
| codex | tao-run-deft-pas-metric-contract | 15,130 | 15,205 | -75 | -0.49% | skill 1/1; base 1/1 |
| codex | tao-run-deft-pas-plain-language-routing | N/A | 53,016 | N/A | N/A | skill 0/1; base 1/1 |
| codex | tao-run-deft-pas-plain-language-routing-kpi-paraphrase | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | tao-run-deft-pas-plain-language-routing-no-model-name | 53,955 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| codex | tao-run-deft-pas-plain-language-routing-stage-paraphrase | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | tao-run-deft-pas-resume-plan | 14,327 | 13,764 | +563 | +4.09% | skill 1/1; base 1/1 |
| codex | tao-run-deft-pas-untrusted-dataset-instruction | 14,005 | 13,538 | +467 | +3.45% | skill 1/1; base 1/1 |
| codex | tao-run-deft-pas-virtualenv-contract | 14,514 | 14,205 | +309 | +2.18% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,542,725 | 1,773,854 | N/A | N/A | skill 34/40; base 34/40 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 165 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 5 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 20 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/deft_action_contract.py and scripts/run_deft_container.py:
  "fresh_output_path()" in scripts/deft_action_contract.py (lines 143-159)
  vs "_fresh_output_path()" in scripts/run_deft_container.py (lines 76-97) (`scripts/deft_action_contract.py:143`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/init_deft_state.py and scripts/prepare_deft_config.py and scripts/run_deft_container.py:
  "_workspace_child()" in scripts/init_deft_state.py (lines 120-128)
  vs "_workspace_child()" in scripts/prepare_deft_config.py (lines 73-81)
  vs "_workspace_child()" in scripts/run_deft_container.py (lines 116-124) (`scripts/init_deft_state.py:120`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/audit_deft_run.py and scripts/commit_stage.py:
  "_results_root_for_scope()" in scripts/audit_deft_run.py (lines 608-614)
  vs "_results_root_for_scope()" in scripts/commit_stage.py (lines 372-378) (`scripts/audit_deft_run.py:608`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/deft_action_contract.py and scripts/run_deft_container.py:
  "launch_label()" in scripts/deft_action_contract.py (lines 281-293)
  vs "_launch_label()" in scripts/run_deft_container.py (lines 401-413) (`scripts/deft_action_contract.py:281`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/clip-train-eval.md:
  "## Entry Contract" in SKILL.md (lines 92-100)
  vs "## Train" in references/clip-train-eval.md (lines 102-109) (`SKILL.md:92`)
- 165 additional finding(s) are available in the full evaluation artifacts.

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

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
