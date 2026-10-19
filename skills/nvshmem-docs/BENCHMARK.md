# Skill Benchmark: nvshmem-docs

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvshmem-docs`
- Evaluation date: 2026-09-16
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 8 evaluation tasks (7 positive, 1 negative)
- Dataset digest: `sha256:b5c07e4c1f1f22740c058d2134cd32b62bf42a16e61e6769cf16319d5239f93d` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 90.4% — baseline ran, but no comparable score was available; uplift unavailable | 86.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 90.0% → 100.0% (+10.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 58.0% → 100.0% (+42.0 points) | 97.5% → 100.0% (+2.5 points) |
| Discoverability | 80.0% — baseline ran, but no comparable score was available; uplift unavailable | 70.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 43.5% → 97.9% (+54.4 points) | 72.2% → 94.2% (+22.0 points) |
| Efficiency | 74.1% — baseline ran, but no comparable score was available; uplift unavailable | 69.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,165,840 | 1,833,706 | N/A | N/A | skill 8/8; base 10/10 |
| claude-code | nvshmem-docs-001 | 147,342 | 362,192 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nvshmem-docs-002 | 147,643 | 30,063 | +117,580 | +391.11% | skill 1/1; base 1/1 |
| claude-code | nvshmem-docs-003 | 152,730 | 62,841 | +89,889 | +143.04% | skill 1/1; base 1/1 |
| claude-code | nvshmem-docs-004 | 30,892 | 30,470 | +422 | +1.38% | skill 1/1; base 1/1 |
| claude-code | nvshmem-docs-005 | 196,690 | 353,741 | -157,051 | -44.40% | skill 1/1; base 1/1 |
| claude-code | nvshmem-docs-006 | 149,979 | 125,953 | +24,026 | +19.08% | skill 1/1; base 1/1 |
| claude-code | nvshmem-docs-007 | 192,831 | 581,274 | -388,443 | -66.83% | skill 1/1; base 1/1 |
| claude-code | nvshmem-docs-008 | 147,733 | 287,172 | -139,439 | -48.56% | skill 1/1; base 1/1 |
| codex | All cases | 760,987 | 264,622 | +496,365 | +187.58% | skill 8/8; base 8/8 |
| codex | nvshmem-docs-001 | 94,182 | 36,112 | +58,070 | +160.81% | skill 1/1; base 1/1 |
| codex | nvshmem-docs-002 | 50,933 | 17,861 | +33,072 | +185.16% | skill 1/1; base 1/1 |
| codex | nvshmem-docs-003 | 110,616 | 32,051 | +78,565 | +245.12% | skill 1/1; base 1/1 |
| codex | nvshmem-docs-004 | 18,404 | 13,803 | +4,601 | +33.33% | skill 1/1; base 1/1 |
| codex | nvshmem-docs-005 | 125,505 | 39,559 | +85,946 | +217.26% | skill 1/1; base 1/1 |
| codex | nvshmem-docs-006 | 111,125 | 53,853 | +57,272 | +106.35% | skill 1/1; base 1/1 |
| codex | nvshmem-docs-007 | 200,202 | 51,598 | +148,604 | +288.00% | skill 1/1; base 1/1 |
| codex | nvshmem-docs-008 | 50,020 | 19,785 | +30,235 | +152.82% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,926,827 | 2,098,328 | N/A | N/A | skill 16/16; base 18/18 |

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
- PII Scan: Scanning 2 files for PII
- Code Integrity & Hygiene: Checking 2 markdown files for dead links
- Unicode Smuggling Detection: No invisible Unicode characters detected in 2 file(s)
- QUALITY: Score: 100.0/100 (Grade: A)
- SCRIPT_LINT: No scripts/ directory found
- Context Deduplication: Collected 2 file(s)
- Inter-Skill Deduplication: Parsed skill 'nvshmem-docs': 161 char description
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
