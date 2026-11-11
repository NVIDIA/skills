# Skill Benchmark: nvshmem-get-started

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvshmem-get-started`
- Evaluation date: 2026-09-17
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 6 evaluation tasks (5 positive, 1 negative)
- Dataset digest: `sha256:d4b763e4ad4d346d9193abfe1de56a613a73b0a613ad065db2291289b139b7a0` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 87.5% — baseline ran, but no comparable score was available; uplift unavailable | 89.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 70.0% → 100.0% (+30.0 points) |
| Correctness | 47.5% → 86.7% (+39.2 points) | 66.0% → 100.0% (+34.0 points) |
| Discoverability | 80.0% — baseline ran, but no comparable score was available; uplift unavailable | 73.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 48.4% → 84.0% (+35.6 points) | 50.0% → 87.8% (+37.8 points) |
| Efficiency | 87.0% — baseline ran, but no comparable score was available; uplift unavailable | 84.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 814,537 | 1,573,572 | N/A | N/A | skill 6/6; base 8/8 |
| claude-code | nvshmem-get-started-001 | 238,094 | 29,967 | +208,127 | +694.52% | skill 1/1; base 1/1 |
| claude-code | nvshmem-get-started-002 | 104,862 | 122,704 | -17,842 | -14.54% | skill 1/1; base 1/1 |
| claude-code | nvshmem-get-started-003 | 66,378 | 64,181 | +2,197 | +3.42% | skill 1/1; base 1/1 |
| claude-code | nvshmem-get-started-004 | 30,620 | 30,442 | +178 | +0.58% | skill 1/1; base 1/1 |
| claude-code | nvshmem-get-started-005 | 152,819 | 437,144 | -284,325 | -65.04% | skill 1/1; base 1/1 |
| claude-code | nvshmem-get-started-006 | 221,764 | 889,134 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 288,184 | 1,310,545 | N/A | N/A | skill 6/6; base 10/10 |
| codex | nvshmem-get-started-001 | 49,824 | 840,216 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nvshmem-get-started-002 | 48,971 | 41,769 | +7,202 | +17.24% | skill 1/1; base 1/1 |
| codex | nvshmem-get-started-003 | 34,353 | 20,781 | +13,572 | +65.31% | skill 1/1; base 1/1 |
| codex | nvshmem-get-started-004 | 24,880 | 18,652 | +6,228 | +33.39% | skill 1/1; base 1/1 |
| codex | nvshmem-get-started-005 | 59,594 | 112,826 | -53,232 | -47.18% | skill 1/1; base 1/1 |
| codex | nvshmem-get-started-006 | 70,562 | 276,301 | N/A | N/A | skill 1/1; base 3/3 |
| ALL AGENTS | Dataset aggregate | 1,102,721 | 2,884,117 | N/A | N/A | skill 12/12; base 18/18 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED** | 11 validator(s); 0 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 6 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- Schema & Repository Governance: Found skill manifest: SKILL.md
- Semantic Version Validation: No semantic version label present; resource will use commit-hash history (opting back out of an existing label is allowed)
- Security Scan: No security vulnerabilities detected (secrets, API keys, credentials)
- PII Scan: Scanning 4 files for PII
- Code Integrity & Hygiene: Checking 4 markdown files for dead links
- Unicode Smuggling Detection: No invisible Unicode characters detected in 4 file(s)
- QUALITY: Score: 100.0/100 (Grade: A)
- SCRIPT_LINT: No scripts/ directory found
- Context Deduplication: Collected 4 file(s)
- Inter-Skill Deduplication: Parsed skill 'nvshmem-get-started': 167 char description
- AGENT_EVAL: Tier 3 evaluation complete: verdict PASS; best agent codex

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
