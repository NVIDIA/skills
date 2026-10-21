# Skill Benchmark: cdf-orchestration-preflight

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cdf-orchestration-preflight`
- Evaluation date: 2026-10-21
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 14 evaluation tasks (10 positive, 4 negative)
- Dataset digest: `sha256:da87380d621af7409004f0f0ec9b73435205584669d91aaed1c756c648cd2978` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 86.8% — baseline ran, but no comparable score was available; uplift unavailable | 79.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 78.6% → 96.4% (+17.8 points) | 67.9% → 89.3% (+21.4 points) |
| Correctness | 12.9% → 80.0% (+67.1 points) | 25.7% → 68.6% (+42.9 points) |
| Discoverability | 99.0% — baseline ran, but no comparable score was available; uplift unavailable | 86.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 36.1% → 68.8% (+32.7 points) | 36.8% → 65.2% (+28.4 points) |
| Efficiency | 89.7% — baseline ran, but no comparable score was available; uplift unavailable | 86.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,910,301 | 3,771,982 | +1,138,319 | +30.18% | skill 14/14; base 14/14 |
| claude-code | audit-before-remediation | 429,948 | 418,695 | +11,253 | +2.69% | skill 1/1; base 1/1 |
| claude-code | av-ffmpeg-sidecar-missing | 558,550 | 153,637 | +404,913 | +263.55% | skill 1/1; base 1/1 |
| claude-code | live-unprofiled-dag | 100,039 | 181,195 | -81,156 | -44.79% | skill 1/1; base 1/1 |
| claude-code | managed-cluster-no-nfs | 330,460 | 395,472 | -65,012 | -16.44% | skill 1/1; base 1/1 |
| claude-code | missing-kubeconfig | 313,561 | 92,412 | +221,149 | +239.31% | skill 1/1; base 1/1 |
| claude-code | negative-author-custom-dag | 673,371 | 218,461 | +454,910 | +208.23% | skill 1/1; base 1/1 |
| claude-code | negative-controller-deploy | 252,517 | 569,000 | -316,483 | -55.62% | skill 1/1; base 1/1 |
| claude-code | negative-trigger-defect-run | 367,770 | 184,501 | +183,269 | +99.33% | skill 1/1; base 1/1 |
| claude-code | negative-workflow-run | 330,531 | 323,045 | +7,486 | +2.32% | skill 1/1; base 1/1 |
| claude-code | osmo-not-proven | 330,719 | 153,187 | +177,532 | +115.89% | skill 1/1; base 1/1 |
| claude-code | partial-one-gpu | 140,617 | 151,573 | -10,956 | -7.23% | skill 1/1; base 1/1 |
| claude-code | prebuilt-controller-image | 388,045 | 498,769 | -110,724 | -22.20% | skill 1/1; base 1/1 |
| claude-code | quick-start-boundary | 363,060 | 181,769 | +181,291 | +99.74% | skill 1/1; base 1/1 |
| claude-code | self-managed-nfs | 331,113 | 250,266 | +80,847 | +32.30% | skill 1/1; base 1/1 |
| codex | All cases | 4,124,743 | 3,714,323 | +410,420 | +11.05% | skill 14/14; base 14/14 |
| codex | audit-before-remediation | 211,148 | 198,388 | +12,760 | +6.43% | skill 1/1; base 1/1 |
| codex | av-ffmpeg-sidecar-missing | 267,741 | 266,473 | +1,268 | +0.48% | skill 1/1; base 1/1 |
| codex | live-unprofiled-dag | 331,552 | 83,750 | +247,802 | +295.88% | skill 1/1; base 1/1 |
| codex | managed-cluster-no-nfs | 268,015 | 186,410 | +81,605 | +43.78% | skill 1/1; base 1/1 |
| codex | missing-kubeconfig | 83,231 | 82,532 | +699 | +0.85% | skill 1/1; base 1/1 |
| codex | negative-author-custom-dag | 516,388 | 197,590 | +318,798 | +161.34% | skill 1/1; base 1/1 |
| codex | negative-controller-deploy | 204,399 | 574,432 | -370,033 | -64.42% | skill 1/1; base 1/1 |
| codex | negative-trigger-defect-run | 596,832 | 1,017,507 | -420,675 | -41.34% | skill 1/1; base 1/1 |
| codex | negative-workflow-run | 472,147 | 324,561 | +147,586 | +45.47% | skill 1/1; base 1/1 |
| codex | osmo-not-proven | 287,581 | 175,458 | +112,123 | +63.90% | skill 1/1; base 1/1 |
| codex | partial-one-gpu | 177,262 | 156,829 | +20,433 | +13.03% | skill 1/1; base 1/1 |
| codex | prebuilt-controller-image | 100,592 | 187,632 | -87,040 | -46.39% | skill 1/1; base 1/1 |
| codex | quick-start-boundary | 173,803 | 115,097 | +58,706 | +51.01% | skill 1/1; base 1/1 |
| codex | self-managed-nfs | 434,052 | 147,664 | +286,388 | +193.95% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 9,035,044 | 7,486,305 | +1,548,739 | +20.69% | skill 28/28; base 28/28 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 50 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 14 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/cdf-orchestration-preflight/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/cdf-orchestration-preflight/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/cdf-orchestration-preflight/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in bootstrap.md (`skills/cdf-orchestration-preflight/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/cdf-orchestration-preflight/SKILL.md`)
- 45 additional finding(s) are available in the full evaluation artifacts.

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
