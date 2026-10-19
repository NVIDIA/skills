# Skill Benchmark: nvshmem-tune-performance

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvshmem-tune-performance`
- Evaluation date: 2026-09-17
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 8 evaluation tasks (7 positive, 1 negative)
- Dataset digest: `sha256:126513e60f7ca3d43627ba9aba60b1359057942392c2730e0118164134b1d05d` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 81.6% — baseline ran, but no comparable score was available; uplift unavailable | 84.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 29.4% → 77.5% (+48.1 points) | 47.7% → 82.5% (+34.8 points) |
| Discoverability | 71.4% — baseline ran, but no comparable score was available; uplift unavailable | 74.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 33.6% → 68.3% (+34.7 points) | 27.1% → 68.1% (+41.0 points) |
| Efficiency | 90.5% — baseline ran, but no comparable score was available; uplift unavailable | 98.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,101,762 | 3,049,342 | N/A | N/A | skill 8/8; base 17/17 |
| claude-code | nvshmem-tune-performance-001 | 97,029 | 280,089 | -183,060 | -65.36% | skill 1/1; base 1/1 |
| claude-code | nvshmem-tune-performance-002 | 161,561 | 930,512 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nvshmem-tune-performance-003 | 192,973 | 579,122 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nvshmem-tune-performance-004 | 197,938 | 343,738 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nvshmem-tune-performance-005 | 196,700 | 534,604 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nvshmem-tune-performance-006 | 162,679 | 319,211 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | nvshmem-tune-performance-007 | 62,185 | 30,045 | +32,140 | +106.97% | skill 1/1; base 1/1 |
| claude-code | nvshmem-tune-performance-008 | 30,697 | 32,021 | -1,324 | -4.13% | skill 1/1; base 1/1 |
| codex | All cases | 280,906 | 1,480,686 | N/A | N/A | skill 8/8; base 13/13 |
| codex | nvshmem-tune-performance-001 | 28,767 | 18,370 | +10,397 | +56.60% | skill 1/1; base 1/1 |
| codex | nvshmem-tune-performance-002 | 43,936 | 211,984 | -168,048 | -79.27% | skill 1/1; base 1/1 |
| codex | nvshmem-tune-performance-003 | 28,626 | 247,663 | N/A | N/A | skill 1/1; base 2/2 |
| codex | nvshmem-tune-performance-004 | 28,687 | 197,900 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nvshmem-tune-performance-005 | 45,143 | 392,058 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nvshmem-tune-performance-006 | 44,053 | 89,216 | -45,163 | -50.62% | skill 1/1; base 1/1 |
| codex | nvshmem-tune-performance-007 | 28,606 | 288,392 | -259,786 | -90.08% | skill 1/1; base 1/1 |
| codex | nvshmem-tune-performance-008 | 33,088 | 35,103 | -2,015 | -5.74% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,382,668 | 4,530,028 | N/A | N/A | skill 16/16; base 30/30 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED** | 11 validator(s); 0 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- Schema & Repository Governance: Found skill manifest: SKILL.md
- Semantic Version Validation: Valid semantic version: 1.0.0
- Security Scan: No security vulnerabilities detected (secrets, API keys, credentials)
- PII Scan: Scanning 1 files for PII
- Code Integrity & Hygiene: Checking 1 markdown files for dead links
- Unicode Smuggling Detection: No invisible Unicode characters detected in 1 file(s)
- QUALITY: Score: 100.0/100 (Grade: A)
- SCRIPT_LINT: No scripts/ directory found
- Context Deduplication: Collected 1 file(s)
- Inter-Skill Deduplication: Parsed skill 'nvshmem-tune-performance': 145 char description
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
