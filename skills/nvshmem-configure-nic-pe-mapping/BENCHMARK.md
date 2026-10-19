# Skill Benchmark: nvshmem-configure-nic-pe-mapping

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvshmem-configure-nic-pe-mapping`
- Evaluation date: 2026-09-18
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 18 evaluation tasks (17 positive, 1 negative)
- Dataset digest: `sha256:3448f01637a5000ccad4f8afc6f67a581949f5b0749b6c973b7c91d24965a8aa` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 90.4% — baseline ran, but no comparable score was available; uplift unavailable | 88.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 96.6% → 100.0% (+3.4 points) | 80.4% → 100.0% (+19.6 points) |
| Correctness | 57.9% → 97.8% (+39.9 points) | 72.9% → 92.2% (+19.3 points) |
| Discoverability | 87.9% — baseline ran, but no comparable score was available; uplift unavailable | 76.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 37.8% → 73.8% (+36.0 points) | 44.0% → 78.5% (+34.5 points) |
| Efficiency | 92.2% — baseline ran, but no comparable score was available; uplift unavailable | 94.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,399,391 | 9,905,108 | N/A | N/A | skill 18/18; base 29/29 |
| claude-code | nvshmem-configure-nic-pe-mapping-001 | 299,936 | 127,444 | +172,492 | +135.35% | skill 1/1; base 1/1 |
| claude-code | nvshmem-configure-nic-pe-mapping-002 | 176,408 | 230,376 | -53,968 | -23.43% | skill 1/1; base 1/1 |
| claude-code | nvshmem-configure-nic-pe-mapping-003 | 386,840 | 1,798,468 | -1,411,628 | -78.49% | skill 1/1; base 1/1 |
| claude-code | nvshmem-configure-nic-pe-mapping-004 | 215,120 | 2,064,784 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nvshmem-configure-nic-pe-mapping-005 | 116,582 | 37,937 | +78,645 | +207.30% | skill 1/1; base 1/1 |
| claude-code | nvshmem-configure-nic-pe-mapping-006 | 113,967 | 715,756 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nvshmem-configure-nic-pe-mapping-007 | 174,466 | 637,897 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nvshmem-configure-nic-pe-mapping-008 | 174,082 | 810,702 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nvshmem-configure-nic-pe-mapping-009 | 117,226 | 107,034 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | nvshmem-configure-nic-pe-mapping-010 | 223,356 | 1,541,978 | -1,318,622 | -85.51% | skill 1/1; base 1/1 |
| claude-code | nvshmem-configure-nic-pe-mapping-011 | 113,886 | 61,321 | +52,565 | +85.72% | skill 1/1; base 1/1 |
| claude-code | nvshmem-configure-nic-pe-mapping-012 | 116,252 | 901,914 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nvshmem-configure-nic-pe-mapping-013 | 218,597 | 133,879 | +84,718 | +63.28% | skill 1/1; base 1/1 |
| claude-code | nvshmem-configure-nic-pe-mapping-014 | 178,706 | 34,518 | +144,188 | +417.72% | skill 1/1; base 1/1 |
| claude-code | nvshmem-configure-nic-pe-mapping-015 | 165,630 | 108,434 | +57,196 | +52.75% | skill 1/1; base 1/1 |
| claude-code | nvshmem-configure-nic-pe-mapping-016 | 171,245 | 34,360 | +136,885 | +398.38% | skill 1/1; base 1/1 |
| claude-code | nvshmem-configure-nic-pe-mapping-017 | 406,346 | 527,576 | -121,230 | -22.98% | skill 1/1; base 1/1 |
| claude-code | nvshmem-configure-nic-pe-mapping-018 | 30,746 | 30,730 | +16 | +0.05% | skill 1/1; base 1/1 |
| codex | All cases | 1,862,245 | 6,068,950 | N/A | N/A | skill 18/18; base 28/28 |
| codex | nvshmem-configure-nic-pe-mapping-001 | 49,947 | 208,142 | -158,195 | -76.00% | skill 1/1; base 1/1 |
| codex | nvshmem-configure-nic-pe-mapping-002 | 49,181 | 574,465 | N/A | N/A | skill 1/1; base 2/2 |
| codex | nvshmem-configure-nic-pe-mapping-003 | 50,293 | 227,315 | -177,022 | -77.88% | skill 1/1; base 1/1 |
| codex | nvshmem-configure-nic-pe-mapping-004 | 155,543 | 1,075,187 | -919,644 | -85.53% | skill 1/1; base 1/1 |
| codex | nvshmem-configure-nic-pe-mapping-005 | 48,972 | 38,717 | +10,255 | +26.49% | skill 1/1; base 1/1 |
| codex | nvshmem-configure-nic-pe-mapping-006 | 69,537 | 176,705 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nvshmem-configure-nic-pe-mapping-007 | 164,624 | 131,140 | +33,484 | +25.53% | skill 1/1; base 1/1 |
| codex | nvshmem-configure-nic-pe-mapping-008 | 117,276 | 115,248 | +2,028 | +1.76% | skill 1/1; base 1/1 |
| codex | nvshmem-configure-nic-pe-mapping-009 | 70,645 | 882,531 | N/A | N/A | skill 1/1; base 2/2 |
| codex | nvshmem-configure-nic-pe-mapping-010 | 116,191 | 152,894 | -36,703 | -24.01% | skill 1/1; base 1/1 |
| codex | nvshmem-configure-nic-pe-mapping-011 | 96,102 | 100,520 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nvshmem-configure-nic-pe-mapping-012 | 110,375 | 802,092 | -691,717 | -86.24% | skill 1/1; base 1/1 |
| codex | nvshmem-configure-nic-pe-mapping-013 | 114,678 | 59,888 | +54,790 | +91.49% | skill 1/1; base 1/1 |
| codex | nvshmem-configure-nic-pe-mapping-014 | 124,527 | 82,405 | +42,122 | +51.12% | skill 1/1; base 1/1 |
| codex | nvshmem-configure-nic-pe-mapping-015 | 112,578 | 375,786 | N/A | N/A | skill 1/1; base 2/2 |
| codex | nvshmem-configure-nic-pe-mapping-016 | 112,801 | 190,047 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nvshmem-configure-nic-pe-mapping-017 | 284,752 | 861,692 | N/A | N/A | skill 1/1; base 2/2 |
| codex | nvshmem-configure-nic-pe-mapping-018 | 14,223 | 14,176 | +47 | +0.33% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 5,261,636 | 15,974,058 | N/A | N/A | skill 36/36; base 57/57 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED** | 11 validator(s); 0 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 18 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- Schema & Repository Governance: Found skill manifest: SKILL.md
- Semantic Version Validation: Valid semantic version: 1.0.0
- Security Scan: No security vulnerabilities detected (secrets, API keys, credentials)
- PII Scan: Scanning 4 files for PII
- Code Integrity & Hygiene: Checking 3 markdown files for dead links
- Unicode Smuggling Detection: No invisible Unicode characters detected in 4 file(s)
- QUALITY: Score: 100.0/100 (Grade: A)
- SCRIPT_LINT: No Python scripts found in scripts/
- Context Deduplication: Collected 4 file(s)
- Inter-Skill Deduplication: Parsed skill 'nvshmem-configure-nic-pe-mapping': 148 char description
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
