# Skill Benchmark: nvshmem-collect-performance-data

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvshmem-collect-performance-data`
- Evaluation date: 2026-09-17
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 12 evaluation tasks (11 positive, 1 negative)
- Dataset digest: `sha256:cd5ae2cd5f82d92864a6f880eed91e5f33a0a07cd670cf1f8fc9731b592f19b4` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 91.1% — baseline ran, but no comparable score was available; uplift unavailable | 87.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 73.7% → 100.0% (+26.3 points) | 85.7% → 100.0% (+14.3 points) |
| Correctness | 62.1% → 100.0% (+37.9 points) | 38.1% → 93.3% (+55.2 points) |
| Discoverability | 89.6% — baseline ran, but no comparable score was available; uplift unavailable | 80.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 47.6% → 89.4% (+41.8 points) | 37.6% → 79.0% (+41.4 points) |
| Efficiency | 76.7% — baseline ran, but no comparable score was available; uplift unavailable | 86.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 5,604,393 | 4,816,411 | N/A | N/A | skill 12/12; base 19/19 |
| claude-code | nvshmem-collect-performance-data-001 | 145,986 | 719,556 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nvshmem-collect-performance-data-002 | 191,528 | 255,303 | -63,775 | -24.98% | skill 1/1; base 1/1 |
| claude-code | nvshmem-collect-performance-data-003 | 842,037 | 319,807 | +522,230 | +163.30% | skill 1/1; base 1/1 |
| claude-code | nvshmem-collect-performance-data-004 | 243,118 | 662,593 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | nvshmem-collect-performance-data-005 | 784,012 | 740,848 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nvshmem-collect-performance-data-006 | 932,091 | 550,217 | +381,874 | +69.40% | skill 1/1; base 1/1 |
| claude-code | nvshmem-collect-performance-data-007 | 893,527 | 444,559 | +448,968 | +100.99% | skill 1/1; base 1/1 |
| claude-code | nvshmem-collect-performance-data-008 | 676,216 | 531,981 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nvshmem-collect-performance-data-009 | 106,323 | 156,216 | -49,893 | -31.94% | skill 1/1; base 1/1 |
| claude-code | nvshmem-collect-performance-data-010 | 368,354 | 183,716 | +184,638 | +100.50% | skill 1/1; base 1/1 |
| claude-code | nvshmem-collect-performance-data-011 | 390,806 | 221,456 | +169,350 | +76.47% | skill 1/1; base 1/1 |
| claude-code | nvshmem-collect-performance-data-012 | 30,395 | 30,159 | +236 | +0.78% | skill 1/1; base 1/1 |
| codex | All cases | 3,195,206 | 4,117,145 | N/A | N/A | skill 12/12; base 21/21 |
| codex | nvshmem-collect-performance-data-001 | 47,681 | 406,880 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nvshmem-collect-performance-data-002 | 508,109 | 173,553 | +334,556 | +192.77% | skill 1/1; base 1/1 |
| codex | nvshmem-collect-performance-data-003 | 492,815 | 374,955 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nvshmem-collect-performance-data-004 | 67,598 | 104,978 | -37,380 | -35.61% | skill 1/1; base 1/1 |
| codex | nvshmem-collect-performance-data-005 | 533,857 | 167,553 | +366,304 | +218.62% | skill 1/1; base 1/1 |
| codex | nvshmem-collect-performance-data-006 | 261,202 | 226,205 | N/A | N/A | skill 1/1; base 2/2 |
| codex | nvshmem-collect-performance-data-007 | 478,738 | 1,345,993 | -867,255 | -64.43% | skill 1/1; base 1/1 |
| codex | nvshmem-collect-performance-data-008 | 304,698 | 353,995 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nvshmem-collect-performance-data-009 | 117,400 | 307,716 | -190,316 | -61.85% | skill 1/1; base 1/1 |
| codex | nvshmem-collect-performance-data-010 | 183,137 | 536,051 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nvshmem-collect-performance-data-011 | 149,606 | 100,304 | +49,302 | +49.15% | skill 1/1; base 1/1 |
| codex | nvshmem-collect-performance-data-012 | 50,365 | 18,962 | +31,403 | +165.61% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 8,799,599 | 8,933,556 | N/A | N/A | skill 24/24; base 40/40 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED** | 11 validator(s); 0 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 12 task(s) |

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
- Inter-Skill Deduplication: Parsed skill 'nvshmem-collect-performance-data': 147 char description
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
