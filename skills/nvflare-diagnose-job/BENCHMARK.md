# Skill Benchmark: nvflare-diagnose-job

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvflare-diagnose-job`
- Evaluation date: 2026-09-16
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 10 evaluation tasks (10 positive)
- Dataset digest: `sha256:d08b09690f9a5de1a62fa97ea64d835ad07beab23ebf5dfaf92c30b245b22c9a` (skill-evaluator-dataset-snapshot/1)
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
| Overall | Not available | 83.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 75.0% → 85.0% (+10.0 points) |
| Correctness | Not available | 66.7% → 94.0% (+27.3 points) |
| Discoverability | Not available | 66.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 79.7% → 89.3% (+9.6 points) |
| Efficiency | Not available | 83.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 12,340,169 | 4,968,401 | N/A | N/A | skill 12/30; base 12/12 |
| claude-code | diagnose-global-negative-web-app | 158,677 | 3,541,027 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | diagnose-negative-create-job | 187,349 | 152,702 | +34,647 | +22.69% | skill 1/1; base 1/1 |
| claude-code | diagnose-negative-download-completed-results | 159,306 | 157,372 | +1,934 | +1.23% | skill 1/1; base 1/1 |
| claude-code | diagnose-negative-healthy-job-lifecycle | 205,143 | 194,804 | +10,339 | +5.31% | skill 1/1; base 1/1 |
| claude-code | diagnose-negative-pytorch-conversion | 10,323,174 | 217,339 | N/A | N/A | skill 3/3; base 1/1 |
| claude-code | diagnose-partial-log-visibility | 308,394 | 184,613 | +123,781 | +67.05% | skill 1/1; base 1/1 |
| claude-code | diagnose-poc-component-not-authorized | 388,384 | 122,981 | +265,403 | +215.81% | skill 1/1; base 1/1 |
| claude-code | diagnose-poisoned-log-content | 216,463 | 122,064 | +94,399 | +77.34% | skill 1/1; base 1/1 |
| claude-code | diagnose-simulation-import-error | 293,959 | 151,190 | +142,769 | +94.43% | skill 1/1; base 1/1 |
| claude-code | diagnose-transfer-progress-timeout | 99,320 | 124,309 | -24,989 | -20.10% | skill 1/1; base 1/1 |
| codex | All cases | 1,613,494 | 1,501,676 | N/A | N/A | skill 10/10; base 12/12 |
| codex | diagnose-global-negative-web-app | 337,813 | 272,296 | +65,517 | +24.06% | skill 1/1; base 1/1 |
| codex | diagnose-negative-create-job | 382,551 | 272,740 | +109,811 | +40.26% | skill 1/1; base 1/1 |
| codex | diagnose-negative-download-completed-results | 100,766 | 285,265 | -184,499 | -64.68% | skill 1/1; base 1/1 |
| codex | diagnose-negative-healthy-job-lifecycle | 187,547 | 128,389 | +59,158 | +46.08% | skill 1/1; base 1/1 |
| codex | diagnose-negative-pytorch-conversion | 132,988 | 70,908 | +62,080 | +87.55% | skill 1/1; base 1/1 |
| codex | diagnose-partial-log-visibility | 176,749 | 195,133 | -18,384 | -9.42% | skill 1/1; base 1/1 |
| codex | diagnose-poc-component-not-authorized | 83,128 | 97,498 | -14,370 | -14.74% | skill 1/1; base 1/1 |
| codex | diagnose-poisoned-log-content | 65,335 | 40,057 | N/A | N/A | skill 1/1; base 3/3 |
| codex | diagnose-simulation-import-error | 100,836 | 70,167 | +30,669 | +43.71% | skill 1/1; base 1/1 |
| codex | diagnose-transfer-progress-timeout | 45,781 | 69,223 | -23,442 | -33.86% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 13,953,663 | 6,470,077 | N/A | N/A | skill 22/40; base 24/24 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 8 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 10 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nvflare-diagnose-job/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nvflare-diagnose-job/SKILL.md`)
- **LOW** QUALITY/quality_correctness: No examples provided (`skills/nvflare-diagnose-job/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (311 chars, recommend 50-150) (`skills/nvflare-diagnose-job/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/nvflare-diagnose-job/SKILL.md`)
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
