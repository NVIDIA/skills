# Skill Benchmark: holoscan-app-debug-build-run

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `holoscan-app-debug-build-run`
- Evaluation date: 2026-10-20
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 5 evaluation tasks (3 positive, 2 negative)
- Dataset digest: `sha256:0d2761be85f364648c039823ecf9f9224d7994dac4d21483df744795dbe7bda0` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 94.6% — baseline ran, but no comparable score was available; uplift unavailable | 76.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 60.0% → 100.0% (+40.0 points) | 60.0% → 76.0% (+16.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 56.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 57.0% → 78.0% (+21.0 points) | 58.0% → 61.8% (+3.8 points) |
| Efficiency | 94.8% — baseline ran, but no comparable score was available; uplift unavailable | 89.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 754,841 | 2,185,333 | -1,430,492 | -65.46% | skill 5/5; base 5/5 |
| claude-code | holoscan-app-debug-build-run-002 | 437,095 | 1,568,126 | -1,131,031 | -72.13% | skill 1/1; base 1/1 |
| claude-code | holoscan-app-debug-build-run-004 | 30,642 | 214,934 | -184,292 | -85.74% | skill 1/1; base 1/1 |
| claude-code | holoscan-app-debug-build-run-007 | 63,153 | 244,385 | -181,232 | -74.16% | skill 1/1; base 1/1 |
| claude-code | holoscan-app-debug-build-run-010 | 118,000 | 35,231 | +82,769 | +234.93% | skill 1/1; base 1/1 |
| claude-code | holoscan-app-debug-build-run-019 | 105,951 | 122,657 | -16,706 | -13.62% | skill 1/1; base 1/1 |
| codex | All cases | 1,092,921 | 316,397 | +776,524 | +245.43% | skill 5/5; base 5/5 |
| codex | holoscan-app-debug-build-run-002 | 904,674 | 146,850 | +757,824 | +516.05% | skill 1/1; base 1/1 |
| codex | holoscan-app-debug-build-run-004 | 13,723 | 41,550 | -27,827 | -66.97% | skill 1/1; base 1/1 |
| codex | holoscan-app-debug-build-run-007 | 129,132 | 99,532 | +29,600 | +29.74% | skill 1/1; base 1/1 |
| codex | holoscan-app-debug-build-run-010 | 14,412 | 14,128 | +284 | +2.01% | skill 1/1; base 1/1 |
| codex | holoscan-app-debug-build-run-019 | 30,980 | 14,337 | +16,643 | +116.08% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,847,762 | 2,501,730 | -653,968 | -26.14% | skill 10/10; base 10/10 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED** | 11 validator(s); 0 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- Schema & Repository Governance: Found skill manifest: SKILL.md
- Semantic Version Validation: No semantic version label present; resource will use commit-hash history (opting back out of an existing label is allowed)
- Security Scan: No security vulnerabilities detected (secrets, API keys, credentials)
- PII Scan: Scanning 5 files for PII
- Code Integrity & Hygiene: Checking 5 markdown files for dead links
- Unicode Smuggling Detection: No invisible Unicode characters detected in 5 file(s)
- QUALITY: Score: 100.0/100 (Grade: A)
- SCRIPT_LINT: No scripts/ directory found
- Context Deduplication: Collected 5 file(s)
- Inter-Skill Deduplication: Parsed skill 'holoscan-app-debug-build-run': 134 char description
- AGENT_EVAL: Tier 3 evaluation complete: verdict PASS; best agent claude-code

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
