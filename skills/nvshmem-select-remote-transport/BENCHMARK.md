# Skill Benchmark: nvshmem-select-remote-transport

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvshmem-select-remote-transport`
- Evaluation date: 2026-09-18
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 14 evaluation tasks (13 positive, 1 negative)
- Dataset digest: `sha256:45a53bc4095635c996256a6ac68469f434e10178cc0f13552333e5d84478c787` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 92.0% — baseline ran, but no comparable score was available; uplift unavailable | 93.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 64.7% → 94.3% (+29.6 points) | 84.3% → 97.1% (+12.8 points) |
| Discoverability | 90.4% — baseline ran, but no comparable score was available; uplift unavailable | 86.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 56.5% → 83.9% (+27.4 points) | 66.8% → 88.2% (+21.4 points) |
| Efficiency | 91.3% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,994,369 | 2,549,164 | N/A | N/A | skill 14/14; base 17/17 |
| claude-code | nvshmem-select-remote-transport-001 | 149,324 | 132,773 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | nvshmem-select-remote-transport-002 | 198,445 | 333,775 | -135,330 | -40.55% | skill 1/1; base 1/1 |
| claude-code | nvshmem-select-remote-transport-003 | 316,621 | 32,501 | +284,120 | +874.19% | skill 1/1; base 1/1 |
| claude-code | nvshmem-select-remote-transport-004 | 173,127 | 103,278 | +69,849 | +67.63% | skill 1/1; base 1/1 |
| claude-code | nvshmem-select-remote-transport-005 | 157,051 | 259,248 | -102,197 | -39.42% | skill 1/1; base 1/1 |
| claude-code | nvshmem-select-remote-transport-006 | 123,366 | 128,197 | -4,831 | -3.77% | skill 1/1; base 1/1 |
| claude-code | nvshmem-select-remote-transport-007 | 119,489 | 169,200 | -49,711 | -29.38% | skill 1/1; base 1/1 |
| claude-code | nvshmem-select-remote-transport-008 | 69,581 | 159,440 | -89,859 | -56.36% | skill 1/1; base 1/1 |
| claude-code | nvshmem-select-remote-transport-009 | 111,687 | 490,076 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | nvshmem-select-remote-transport-010 | 125,563 | 162,233 | -36,670 | -22.60% | skill 1/1; base 1/1 |
| claude-code | nvshmem-select-remote-transport-011 | 69,957 | 228,839 | -158,882 | -69.43% | skill 1/1; base 1/1 |
| claude-code | nvshmem-select-remote-transport-012 | 229,138 | 192,589 | +36,549 | +18.98% | skill 1/1; base 1/1 |
| claude-code | nvshmem-select-remote-transport-013 | 119,631 | 126,910 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | nvshmem-select-remote-transport-014 | 31,389 | 30,105 | +1,284 | +4.27% | skill 1/1; base 1/1 |
| codex | All cases | 1,515,717 | 1,260,373 | +255,344 | +20.26% | skill 14/14; base 14/14 |
| codex | nvshmem-select-remote-transport-001 | 121,228 | 90,125 | +31,103 | +34.51% | skill 1/1; base 1/1 |
| codex | nvshmem-select-remote-transport-002 | 149,314 | 79,524 | +69,790 | +87.76% | skill 1/1; base 1/1 |
| codex | nvshmem-select-remote-transport-003 | 123,882 | 99,430 | +24,452 | +24.59% | skill 1/1; base 1/1 |
| codex | nvshmem-select-remote-transport-004 | 71,020 | 134,450 | -63,430 | -47.18% | skill 1/1; base 1/1 |
| codex | nvshmem-select-remote-transport-005 | 122,327 | 102,622 | +19,705 | +19.20% | skill 1/1; base 1/1 |
| codex | nvshmem-select-remote-transport-006 | 160,379 | 96,029 | +64,350 | +67.01% | skill 1/1; base 1/1 |
| codex | nvshmem-select-remote-transport-007 | 63,266 | 83,119 | -19,853 | -23.89% | skill 1/1; base 1/1 |
| codex | nvshmem-select-remote-transport-008 | 105,314 | 71,321 | +33,993 | +47.66% | skill 1/1; base 1/1 |
| codex | nvshmem-select-remote-transport-009 | 88,699 | 130,655 | -41,956 | -32.11% | skill 1/1; base 1/1 |
| codex | nvshmem-select-remote-transport-010 | 123,643 | 71,687 | +51,956 | +72.48% | skill 1/1; base 1/1 |
| codex | nvshmem-select-remote-transport-011 | 106,298 | 82,334 | +23,964 | +29.11% | skill 1/1; base 1/1 |
| codex | nvshmem-select-remote-transport-012 | 170,010 | 73,783 | +96,227 | +130.42% | skill 1/1; base 1/1 |
| codex | nvshmem-select-remote-transport-013 | 96,289 | 131,348 | -35,059 | -26.69% | skill 1/1; base 1/1 |
| codex | nvshmem-select-remote-transport-014 | 14,048 | 13,946 | +102 | +0.73% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 3,510,086 | 3,809,537 | N/A | N/A | skill 28/28; base 31/31 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED** | 11 validator(s); 0 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 14 task(s) |

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
- Inter-Skill Deduplication: Parsed skill 'nvshmem-select-remote-transport': 144 char description
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
