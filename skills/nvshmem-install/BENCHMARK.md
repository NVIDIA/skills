# Skill Benchmark: nvshmem-install

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvshmem-install`
- Evaluation date: 2026-09-18
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 8 evaluation tasks (7 positive, 1 negative)
- Dataset digest: `sha256:aacad0c211dc8fa408a75eb047468fb537abb3a6201640ec4145164ebe0d48b0` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 90.3% — baseline ran, but no comparable score was available; uplift unavailable | 88.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 85.0% → 93.8% (+8.8 points) |
| Correctness | 90.0% → 100.0% (+10.0 points) | 94.0% → 100.0% (+6.0 points) |
| Discoverability | 82.9% — baseline ran, but no comparable score was available; uplift unavailable | 88.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 80.8% → 98.3% (+17.5 points) | 55.4% → 93.3% (+37.9 points) |
| Efficiency | 70.5% — baseline ran, but no comparable score was available; uplift unavailable | 68.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,381,933 | 943,506 | +2,438,427 | +258.44% | skill 8/8; base 8/8 |
| claude-code | nvshmem-install-001 | 434,791 | 146,647 | +288,144 | +196.49% | skill 1/1; base 1/1 |
| claude-code | nvshmem-install-002 | 573,510 | 79,764 | +493,746 | +619.01% | skill 1/1; base 1/1 |
| claude-code | nvshmem-install-003 | 520,763 | 138,713 | +382,050 | +275.42% | skill 1/1; base 1/1 |
| claude-code | nvshmem-install-004 | 29,906 | 29,836 | +70 | +0.23% | skill 1/1; base 1/1 |
| claude-code | nvshmem-install-005 | 793,985 | 97,379 | +696,606 | +715.36% | skill 1/1; base 1/1 |
| claude-code | nvshmem-install-006 | 713,764 | 160,984 | +552,780 | +343.38% | skill 1/1; base 1/1 |
| claude-code | nvshmem-install-007 | 249,701 | 165,637 | +84,064 | +50.75% | skill 1/1; base 1/1 |
| claude-code | nvshmem-install-008 | 65,513 | 124,546 | -59,033 | -47.40% | skill 1/1; base 1/1 |
| codex | All cases | 2,034,302 | 1,075,328 | N/A | N/A | skill 8/8; base 10/10 |
| codex | nvshmem-install-001 | 68,410 | 18,735 | +49,675 | +265.15% | skill 1/1; base 1/1 |
| codex | nvshmem-install-002 | 89,634 | 32,846 | +56,788 | +172.89% | skill 1/1; base 1/1 |
| codex | nvshmem-install-003 | 239,102 | 26,858 | +212,244 | +790.24% | skill 1/1; base 1/1 |
| codex | nvshmem-install-004 | 13,687 | 13,584 | +103 | +0.76% | skill 1/1; base 1/1 |
| codex | nvshmem-install-005 | 1,397,359 | 67,129 | +1,330,230 | +1981.60% | skill 1/1; base 1/1 |
| codex | nvshmem-install-006 | 52,556 | 17,746 | +34,810 | +196.16% | skill 1/1; base 1/1 |
| codex | nvshmem-install-007 | 125,640 | 80,846 | +44,794 | +55.41% | skill 1/1; base 1/1 |
| codex | nvshmem-install-008 | 47,914 | 817,584 | N/A | N/A | skill 1/1; base 3/3 |
| ALL AGENTS | Dataset aggregate | 5,416,235 | 2,018,834 | N/A | N/A | skill 16/16; base 18/18 |

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
- PII Scan: Scanning 6 files for PII
- Code Integrity & Hygiene: Checking 5 markdown files for dead links
- Unicode Smuggling Detection: No invisible Unicode characters detected in 6 file(s)
- QUALITY: Score: 100.0/100 (Grade: A)
- SCRIPT_LINT: No Python scripts found in scripts/
- Context Deduplication: Collected 6 file(s)
- Inter-Skill Deduplication: Parsed skill 'nvshmem-install': 106 char description
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
