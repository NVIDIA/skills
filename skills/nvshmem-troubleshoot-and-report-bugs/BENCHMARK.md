# Skill Benchmark: nvshmem-troubleshoot-and-report-bugs

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvshmem-troubleshoot-and-report-bugs`
- Evaluation date: 2026-09-17
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (3 positive, 1 negative)
- Dataset digest: `sha256:921c8be88893fc6098dc52bb100827d2f0110f99530aa7765eff3e8d7cfe7c72` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 92.3% — baseline ran, but no comparable score was available; uplift unavailable | 89.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 70.0% → 95.0% (+25.0 points) | 70.0% → 90.0% (+20.0 points) |
| Discoverability | 90.0% — baseline ran, but no comparable score was available; uplift unavailable | 90.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 53.3% → 80.0% (+26.7 points) | 56.3% → 85.0% (+28.7 points) |
| Efficiency | 96.7% — baseline ran, but no comparable score was available; uplift unavailable | 81.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 328,443 | 408,580 | N/A | N/A | skill 4/4; base 6/6 |
| claude-code | nvshmem-troubleshoot-and-report-bugs-001 | 114,852 | 217,701 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nvshmem-troubleshoot-and-report-bugs-002 | 118,199 | 95,815 | +22,384 | +23.36% | skill 1/1; base 1/1 |
| claude-code | nvshmem-troubleshoot-and-report-bugs-003 | 65,168 | 65,032 | +136 | +0.21% | skill 1/1; base 1/1 |
| claude-code | nvshmem-troubleshoot-and-report-bugs-004 | 30,224 | 30,032 | +192 | +0.64% | skill 1/1; base 1/1 |
| codex | All cases | 222,919 | 166,217 | N/A | N/A | skill 4/4; base 6/6 |
| codex | nvshmem-troubleshoot-and-report-bugs-001 | 67,367 | 67,264 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nvshmem-troubleshoot-and-report-bugs-002 | 87,511 | 37,961 | +49,550 | +130.53% | skill 1/1; base 1/1 |
| codex | nvshmem-troubleshoot-and-report-bugs-003 | 54,310 | 47,285 | +7,025 | +14.86% | skill 1/1; base 1/1 |
| codex | nvshmem-troubleshoot-and-report-bugs-004 | 13,731 | 13,707 | +24 | +0.18% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 551,362 | 574,797 | N/A | N/A | skill 8/8; base 12/12 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED** | 11 validator(s); 0 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- Schema & Repository Governance: Found skill manifest: SKILL.md
- Semantic Version Validation: Valid semantic version: 1.0.0
- Security Scan: No security vulnerabilities detected (secrets, API keys, credentials)
- PII Scan: Scanning 3 files for PII
- Code Integrity & Hygiene: Checking 3 markdown files for dead links
- Unicode Smuggling Detection: No invisible Unicode characters detected in 3 file(s)
- QUALITY: Score: 100.0/100 (Grade: A)
- SCRIPT_LINT: No scripts/ directory found
- Context Deduplication: Collected 3 file(s)
- Inter-Skill Deduplication: Parsed skill 'nvshmem-troubleshoot-and-report-bugs': 129 char description
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
