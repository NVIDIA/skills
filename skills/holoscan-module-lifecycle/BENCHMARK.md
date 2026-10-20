# Skill Benchmark: holoscan-module-lifecycle

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `holoscan-module-lifecycle`
- Evaluation date: 2026-10-20
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 8 evaluation tasks (7 positive, 1 negative)
- Dataset digest: `sha256:e4d70204e72ae37b649556e34cec480522ea05caa65ad65f6e267f4cbfc30ef5` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 95.1% — baseline ran, but no comparable score was available; uplift unavailable | 82.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 87.5% → 100.0% (+12.5 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 67.5% → 100.0% (+32.5 points) | 55.0% → 80.0% (+25.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 78.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 48.9% → 82.0% (+33.1 points) | 51.8% → 68.9% (+17.1 points) |
| Efficiency | 93.4% — baseline ran, but no comparable score was available; uplift unavailable | 86.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,471,595 | 3,485,627 | -2,014,032 | -57.78% | skill 8/8; base 8/8 |
| claude-code | holoscan-module-lifecycle-005 | 663,660 | 1,737,070 | -1,073,410 | -61.79% | skill 1/1; base 1/1 |
| claude-code | holoscan-module-lifecycle-006 | 115,764 | 94,110 | +21,654 | +23.01% | skill 1/1; base 1/1 |
| claude-code | holoscan-module-lifecycle-008 | 116,767 | 915,082 | -798,315 | -87.24% | skill 1/1; base 1/1 |
| claude-code | holoscan-module-lifecycle-009 | 146,427 | 187,252 | -40,825 | -21.80% | skill 1/1; base 1/1 |
| claude-code | holoscan-module-lifecycle-010 | 111,872 | 235,986 | -124,114 | -52.59% | skill 1/1; base 1/1 |
| claude-code | holoscan-module-lifecycle-011 | 107,150 | 125,336 | -18,186 | -14.51% | skill 1/1; base 1/1 |
| claude-code | holoscan-module-lifecycle-012 | 107,631 | 64,810 | +42,821 | +66.07% | skill 1/1; base 1/1 |
| claude-code | holoscan-module-lifecycle-013 | 102,324 | 125,981 | -23,657 | -18.78% | skill 1/1; base 1/1 |
| codex | All cases | 532,333 | 452,365 | +79,968 | +17.68% | skill 8/8; base 8/8 |
| codex | holoscan-module-lifecycle-005 | 127,752 | 129,909 | -2,157 | -1.66% | skill 1/1; base 1/1 |
| codex | holoscan-module-lifecycle-006 | 14,238 | 13,979 | +259 | +1.85% | skill 1/1; base 1/1 |
| codex | holoscan-module-lifecycle-008 | 98,622 | 55,373 | +43,249 | +78.10% | skill 1/1; base 1/1 |
| codex | holoscan-module-lifecycle-009 | 115,421 | 59,213 | +56,208 | +94.93% | skill 1/1; base 1/1 |
| codex | holoscan-module-lifecycle-010 | 89,627 | 118,837 | -29,210 | -24.58% | skill 1/1; base 1/1 |
| codex | holoscan-module-lifecycle-011 | 30,225 | 24,962 | +5,263 | +21.08% | skill 1/1; base 1/1 |
| codex | holoscan-module-lifecycle-012 | 41,893 | 35,744 | +6,149 | +17.20% | skill 1/1; base 1/1 |
| codex | holoscan-module-lifecycle-013 | 14,555 | 14,348 | +207 | +1.44% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,003,928 | 3,937,992 | -1,934,064 | -49.11% | skill 16/16; base 16/16 |

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
- Semantic Version Validation: No semantic version label present; resource will use commit-hash history (opting back out of an existing label is allowed)
- Security Scan: No security vulnerabilities detected (secrets, API keys, credentials)
- PII Scan: Scanning 5 files for PII
- Code Integrity & Hygiene: Checking 5 markdown files for dead links
- Unicode Smuggling Detection: No invisible Unicode characters detected in 5 file(s)
- QUALITY: Score: 100.0/100 (Grade: A)
- SCRIPT_LINT: No scripts/ directory found
- Context Deduplication: Collected 5 file(s)
- Inter-Skill Deduplication: Parsed skill 'holoscan-module-lifecycle': 174 char description
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
