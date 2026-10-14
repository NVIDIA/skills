# Skill Benchmark: parabricks

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `parabricks`
- Evaluation date: 2026-10-13
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 18 evaluation tasks (17 positive, 1 negative)
- Dataset digest: `sha256:17ca68b3af6d736e9eca9d4b35718991af9564ec9445ac2796bce041668c9567` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 93.8% — baseline ran, but no comparable score was available; uplift unavailable | 92.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 94.4% → 88.9% (-5.5 points) | 91.7% → 100.0% (+8.3 points) |
| Correctness | 94.4% → 98.9% (+4.5 points) | 97.8% → 100.0% (+2.2 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 87.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 76.5% → 91.8% (+15.3 points) | 83.3% → 91.9% (+8.6 points) |
| Efficiency | 89.5% — baseline ran, but no comparable score was available; uplift unavailable | 84.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,381,424 | 6,702,748 | -2,321,324 | -34.63% | skill 18/18; base 18/18 |
| claude-code | parabricks-001 | 142,693 | 97,867 | +44,826 | +45.80% | skill 1/1; base 1/1 |
| claude-code | parabricks-002 | 107,565 | 32,944 | +74,621 | +226.51% | skill 1/1; base 1/1 |
| claude-code | parabricks-003 | 106,801 | 35,928 | +70,873 | +197.26% | skill 1/1; base 1/1 |
| claude-code | parabricks-004 | 121,495 | 89,131 | +32,364 | +36.31% | skill 1/1; base 1/1 |
| claude-code | parabricks-005 | 327,340 | 126,284 | +201,056 | +159.21% | skill 1/1; base 1/1 |
| claude-code | parabricks-006 | 114,122 | 34,794 | +79,328 | +227.99% | skill 1/1; base 1/1 |
| claude-code | parabricks-007 | 98,232 | 583,042 | -484,810 | -83.15% | skill 1/1; base 1/1 |
| claude-code | parabricks-008 | 146,435 | 32,324 | +114,111 | +353.02% | skill 1/1; base 1/1 |
| claude-code | parabricks-009 | 151,739 | 40,018 | +111,721 | +279.18% | skill 1/1; base 1/1 |
| claude-code | parabricks-010 | 107,999 | 36,647 | +71,352 | +194.70% | skill 1/1; base 1/1 |
| claude-code | parabricks-011 | 109,750 | 464,258 | -354,508 | -76.36% | skill 1/1; base 1/1 |
| claude-code | parabricks-012 | 104,353 | 496,373 | -392,020 | -78.98% | skill 1/1; base 1/1 |
| claude-code | parabricks-013 | 176,319 | 198,676 | -22,357 | -11.25% | skill 1/1; base 1/1 |
| claude-code | parabricks-014 | 146,954 | 50,434 | +96,520 | +191.38% | skill 1/1; base 1/1 |
| claude-code | parabricks-015 | 188,798 | 40,088 | +148,710 | +370.96% | skill 1/1; base 1/1 |
| claude-code | parabricks-016 | 755,795 | 1,583,044 | -827,249 | -52.26% | skill 1/1; base 1/1 |
| claude-code | parabricks-017 | 1,186,237 | 1,278,696 | -92,459 | -7.23% | skill 1/1; base 1/1 |
| claude-code | parabricks-018 | 288,797 | 1,482,200 | -1,193,403 | -80.52% | skill 1/1; base 1/1 |
| codex | All cases | 1,904,066 | 1,118,265 | +785,801 | +70.27% | skill 18/18; base 18/18 |
| codex | parabricks-001 | 263,177 | 104,558 | +158,619 | +151.70% | skill 1/1; base 1/1 |
| codex | parabricks-002 | 150,198 | 30,087 | +120,111 | +399.21% | skill 1/1; base 1/1 |
| codex | parabricks-003 | 159,741 | 53,820 | +105,921 | +196.81% | skill 1/1; base 1/1 |
| codex | parabricks-004 | 18,017 | 17,995 | +22 | +0.12% | skill 1/1; base 1/1 |
| codex | parabricks-005 | 110,503 | 32,181 | +78,322 | +243.38% | skill 1/1; base 1/1 |
| codex | parabricks-006 | 50,986 | 54,054 | -3,068 | -5.68% | skill 1/1; base 1/1 |
| codex | parabricks-007 | 48,819 | 23,382 | +25,437 | +108.79% | skill 1/1; base 1/1 |
| codex | parabricks-008 | 70,936 | 33,757 | +37,179 | +110.14% | skill 1/1; base 1/1 |
| codex | parabricks-009 | 69,630 | 39,964 | +29,666 | +74.23% | skill 1/1; base 1/1 |
| codex | parabricks-010 | 100,094 | 31,157 | +68,937 | +221.26% | skill 1/1; base 1/1 |
| codex | parabricks-011 | 25,515 | 27,485 | -1,970 | -7.17% | skill 1/1; base 1/1 |
| codex | parabricks-012 | 49,343 | 30,529 | +18,814 | +61.63% | skill 1/1; base 1/1 |
| codex | parabricks-013 | 120,873 | 25,598 | +95,275 | +372.20% | skill 1/1; base 1/1 |
| codex | parabricks-014 | 55,186 | 21,556 | +33,630 | +156.01% | skill 1/1; base 1/1 |
| codex | parabricks-015 | 91,439 | 47,958 | +43,481 | +90.66% | skill 1/1; base 1/1 |
| codex | parabricks-016 | 277,386 | 251,731 | +25,655 | +10.19% | skill 1/1; base 1/1 |
| codex | parabricks-017 | 99,179 | 241,529 | -142,350 | -58.94% | skill 1/1; base 1/1 |
| codex | parabricks-018 | 143,044 | 50,924 | +92,120 | +180.90% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 6,285,490 | 7,821,013 | -1,535,523 | -19.63% | skill 36/36; base 36/36 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 11 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 18 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/bionemo-agent-toolkit/skills/parabricks/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in pbrun-applybqsr.md (`skills/bionemo-agent-toolkit/skills/parabricks/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/bionemo-agent-toolkit/skills/parabricks`)
- **MEDIUM** SECURITY/Unknown (LP3): MCP Least Privilege: The skill declares capabilities involving file_read, network access, and shell execution (via bundled scripts like check (`SKILL.md:1`)
- **MEDIUM** SECURITY/Unknown (RP1): MCP Rug Pull: Docker image references without a specific tag (:latest is implicit) or digest (@sha256:...) can be silently replaced by (`references/command-conventions.md:23`)
- 6 additional finding(s) are available in the full evaluation artifacts.

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
