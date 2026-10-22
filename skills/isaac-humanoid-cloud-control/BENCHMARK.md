# Skill Benchmark: isaac-humanoid-cloud-control

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `isaac-humanoid-cloud-control`
- Evaluation date: 2026-10-22
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-sonnet-5-5`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 15 evaluation tasks (15 positive)
- Dataset digest: `sha256:e864e89308afa62c8f968b3061cb82e668f628b21b15e26ce6c91b79f3bbe565` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 82.3% — baseline ran, but no comparable score was available; uplift unavailable | 68.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 80.0% → 100.0% (+20.0 points) | 46.7% → 60.0% (+13.3 points) |
| Correctness | 28.0% → 78.7% (+50.7 points) | 20.0% → 72.0% (+52.0 points) |
| Discoverability | 92.3% — baseline ran, but no comparable score was available; uplift unavailable | 80.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 30.7% → 51.8% (+21.1 points) | 24.7% → 51.0% (+26.3 points) |
| Efficiency | 88.5% — baseline ran, but no comparable score was available; uplift unavailable | 81.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,174,730 | 1,895,234 | +2,279,496 | +120.28% | skill 15/15; base 15/15 |
| claude-code | isaac-humanoid-cloud-control-001 | 229,766 | 125,065 | +104,701 | +83.72% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-002 | 210,822 | 89,316 | +121,506 | +136.04% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-003 | 269,030 | 126,576 | +142,454 | +112.54% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-004 | 100,935 | 124,884 | -23,949 | -19.18% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-005 | 221,824 | 89,936 | +131,888 | +146.65% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-006 | 422,256 | 191,430 | +230,826 | +120.58% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-007 | 376,010 | 163,139 | +212,871 | +130.48% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-008 | 283,805 | 125,098 | +158,707 | +126.87% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-009 | 283,647 | 93,850 | +189,797 | +202.23% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-010 | 359,187 | 121,349 | +237,838 | +196.00% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-011 | 319,573 | 153,737 | +165,836 | +107.87% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-012 | 375,878 | 121,381 | +254,497 | +209.67% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-013 | 236,760 | 155,443 | +81,317 | +52.31% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-release-retry | 108,712 | 90,730 | +17,982 | +19.82% | skill 1/1; base 1/1 |
| claude-code | isaac-humanoid-cloud-control-security-boundary | 376,525 | 123,300 | +253,225 | +205.37% | skill 1/1; base 1/1 |
| codex | All cases | 6,134,063 | 5,829,823 | +304,240 | +5.22% | skill 15/15; base 15/15 |
| codex | isaac-humanoid-cloud-control-001 | 564,812 | 147,039 | +417,773 | +284.12% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-002 | 606,805 | 13,318 | +593,487 | +4456.28% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-003 | 202,340 | 148,091 | +54,249 | +36.63% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-004 | 78,392 | 265,817 | -187,425 | -70.51% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-005 | 221,298 | 97,537 | +123,761 | +126.89% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-006 | 745,145 | 983,567 | -238,422 | -24.24% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-007 | 902,032 | 1,052,456 | -150,424 | -14.29% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-008 | 141,097 | 393,603 | -252,506 | -64.15% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-009 | 163,886 | 151,463 | +12,423 | +8.20% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-010 | 634,668 | 409,409 | +225,259 | +55.02% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-011 | 331,993 | 1,439,776 | -1,107,783 | -76.94% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-012 | 480,119 | 101,151 | +378,968 | +374.66% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-013 | 577,492 | 157,192 | +420,300 | +267.38% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-release-retry | 82,709 | 68,858 | +13,851 | +20.12% | skill 1/1; base 1/1 |
| codex | isaac-humanoid-cloud-control-security-boundary | 401,275 | 400,546 | +729 | +0.18% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 10,308,793 | 7,725,057 | +2,583,736 | +33.45% | skill 30/30; base 30/30 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 58 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 2 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 15 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/ordered-startup.md and references/release-4.6.md:
  "# Ordered startup and mission workflow" in references/ordered-startup.md (lines 162-165)
  vs "## Isaac Sim navigation" in references/release-4.6.md (lines 84-87) (`references/ordered-startup.md:162`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/ordered-startup.md and references/release-4.6.md:
  "# Ordered startup and mission workflow" in references/ordered-startup.md (lines 142-161)
  vs "## Isaac Sim navigation" in references/release-4.6.md (lines 76-83) (`references/ordered-startup.md:142`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in dependencies.md (`skills/isaac-humanoid-cloud-control/SKILL.md`)
- **MEDIUM** SCRIPT_LINT/flat_script: simulation_identity.py has no function definitions (flat script) (`skills/isaac-humanoid-cloud-control/scripts/simulation_identity.py`)
- **MEDIUM** SECURITY/Unknown (LP3): MCP Least Privilege: Without declared permissions the skill's intent is opaque and cannot be validated. (`SKILL.md:1`)
- 55 additional finding(s) are available in the full evaluation artifacts.

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
