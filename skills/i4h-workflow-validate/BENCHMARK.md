# Skill Benchmark: i4h-workflow-validate

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `i4h-workflow-validate`
- Evaluation date: 2026-10-01
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 20 evaluation tasks (20 positive)
- Dataset digest: `sha256:7edc28d803fd2c43a71c294c0692be370df7cb366c30d019a04cae5ee0ff15d2` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 73.0% — baseline ran, but no comparable score was available; uplift unavailable | 58.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 90.0% → 80.0% (-10.0 points) | 25.0% → 22.5% (-2.5 points) |
| Correctness | 4.0% → 88.0% (+84.0 points) | 63.0% → 91.0% (+28.0 points) |
| Discoverability | 94.8% — baseline ran, but no comparable score was available; uplift unavailable | 80.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 2.7% → 26.2% (+23.5 points) | 24.4% → 42.8% (+18.4 points) |
| Efficiency | 75.9% — baseline ran, but no comparable score was available; uplift unavailable | 55.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 11,228,344 | 6,977,680 | N/A | N/A | skill 6/20; base 6/20 |
| claude-code | i4h-workflow-validate-readme-new-env-rule-based-record | 318,822 | 321,012 | -2,190 | -0.68% | skill 1/1; base 1/1 |
| claude-code | i4h-workflow-validate-readme-quick-locomanip-cart | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | i4h-workflow-validate-readme-quick-locomanip-tray | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | i4h-workflow-validate-readme-quick-new-checkpoint | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | i4h-workflow-validate-readme-quick-scissor-two | 4,522,475 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| claude-code | i4h-workflow-validate-readme-quick-surgical-lift-needle-rule-based | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | i4h-workflow-validate-readme-quick-surgical-reach-newton | N/A | 3,260,419 | N/A | N/A | skill 0/1; base 1/1 |
| claude-code | i4h-workflow-validate-readme-quick-surgical-reach-rule-based | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | i4h-workflow-validate-readme-quick-trocar | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | i4h-workflow-validate-readme-quick-ultrasound | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | i4h-workflow-validate-readme-robotic-instrument-reach | 207,894 | 447,313 | -239,419 | -53.52% | skill 1/1; base 1/1 |
| claude-code | i4h-workflow-validate-specialty-endoluminal-demo | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | i4h-workflow-validate-specialty-endoluminal-total-segmentator | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | i4h-workflow-validate-specialty-scissor-policy | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | i4h-workflow-validate-specialty-surgical-dual-psm | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | i4h-workflow-validate-specialty-surgical-lift-block | 5,555,783 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| claude-code | i4h-workflow-validate-specialty-surgical-lift-needle-organs | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| claude-code | i4h-workflow-validate-specialty-surgical-star | N/A | 1,973,749 | N/A | N/A | skill 0/1; base 1/1 |
| claude-code | i4h-workflow-validate-table-scissor-three | 375,917 | 280,652 | +95,265 | +33.94% | skill 1/1; base 1/1 |
| claude-code | i4h-workflow-validate-ultrasound-probe-reach | 247,453 | 694,535 | -447,082 | -64.37% | skill 1/1; base 1/1 |
| codex | All cases | 31,730,638 | 41,836,769 | N/A | N/A | skill 6/20; base 6/20 |
| codex | i4h-workflow-validate-readme-new-env-rule-based-record | 440,339 | 3,329,185 | -2,888,846 | -86.77% | skill 1/1; base 1/1 |
| codex | i4h-workflow-validate-readme-quick-locomanip-cart | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | i4h-workflow-validate-readme-quick-locomanip-tray | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | i4h-workflow-validate-readme-quick-new-checkpoint | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | i4h-workflow-validate-readme-quick-scissor-two | 14,773,729 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| codex | i4h-workflow-validate-readme-quick-surgical-lift-needle-rule-based | N/A | 15,747,712 | N/A | N/A | skill 0/1; base 1/1 |
| codex | i4h-workflow-validate-readme-quick-surgical-reach-newton | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | i4h-workflow-validate-readme-quick-surgical-reach-rule-based | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | i4h-workflow-validate-readme-quick-trocar | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | i4h-workflow-validate-readme-quick-ultrasound | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | i4h-workflow-validate-readme-robotic-instrument-reach | 410,174 | 590,738 | -180,564 | -30.57% | skill 1/1; base 1/1 |
| codex | i4h-workflow-validate-specialty-endoluminal-demo | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | i4h-workflow-validate-specialty-endoluminal-total-segmentator | 13,249,921 | N/A | N/A | N/A | skill 1/1; base 0/1 |
| codex | i4h-workflow-validate-specialty-scissor-policy | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | i4h-workflow-validate-specialty-surgical-dual-psm | N/A | 17,174,793 | N/A | N/A | skill 0/1; base 1/1 |
| codex | i4h-workflow-validate-specialty-surgical-lift-block | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | i4h-workflow-validate-specialty-surgical-lift-needle-organs | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | i4h-workflow-validate-specialty-surgical-star | N/A | N/A | N/A | N/A | skill 0/1; base 0/1 |
| codex | i4h-workflow-validate-table-scissor-three | 2,067,679 | 3,561,756 | -1,494,077 | -41.95% | skill 1/1; base 1/1 |
| codex | i4h-workflow-validate-ultrasound-probe-reach | 788,796 | 1,432,585 | -643,789 | -44.94% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 42,958,982 | 48,814,449 | N/A | N/A | skill 12/40; base 12/40 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 3 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 20 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **LOW** QUALITY/quality_reliability: Inputs are used but no dedicated Inputs section is documented (`team-skills/holoscan/i4h-workflows/i4h-workflow-validate/SKILL.md`)
- **LOW** QUALITY/quality_reliability: Structured output is used but no dedicated Output Format section is documented (`team-skills/holoscan/i4h-workflows/i4h-workflow-validate/SKILL.md`)
- **LOW** SECURITY/Unknown (SQP-2): The skill silently clones a remote repository into the user's home directory ($HOME/$I4H_REPO_DIR_NAME) without explicit (`SKILL.md:40`)

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
