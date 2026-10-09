# Skill Benchmark: codonfm-finetune

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `codonfm-finetune`
- Evaluation date: 2026-10-07
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 3 evaluation tasks (3 positive)
- Dataset digest: `sha256:81d6d44379cdd6e73149247223022a3b07d3b00517da29ff30435ddc7aedc045` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 85.0% — baseline ran, but no comparable score was available; uplift unavailable | 82.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 66.7% (-33.3 points) | 33.3% → 66.7% (+33.4 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 91.7% — baseline ran, but no comparable score was available; uplift unavailable | 76.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 86.7% → 86.7% (±0.0 points) | 96.7% → 90.0% (-6.7 points) |
| Efficiency | 80.0% — baseline ran, but no comparable score was available; uplift unavailable | 80.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,259,711 | 5,318,079 | -3,058,368 | -57.51% | skill 3/3; base 3/3 |
| claude-code | codonfm-finetune-001 | 975,823 | 2,626,474 | -1,650,651 | -62.85% | skill 1/1; base 1/1 |
| claude-code | codonfm-finetune-002 | 728,893 | 1,887,306 | -1,158,413 | -61.38% | skill 1/1; base 1/1 |
| claude-code | codonfm-finetune-003 | 554,995 | 804,299 | -249,304 | -31.00% | skill 1/1; base 1/1 |
| codex | All cases | 1,217,895 | 1,719,144 | -501,249 | -29.16% | skill 3/3; base 3/3 |
| codex | codonfm-finetune-001 | 536,353 | 1,008,386 | -472,033 | -46.81% | skill 1/1; base 1/1 |
| codex | codonfm-finetune-002 | 435,825 | 405,824 | +30,001 | +7.39% | skill 1/1; base 1/1 |
| codex | codonfm-finetune-003 | 245,717 | 304,934 | -59,217 | -19.42% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 3,477,606 | 7,037,223 | -3,559,617 | -50.58% | skill 6/6; base 6/6 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 16 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 3 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/codonfm-finetune/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/codonfm-finetune/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/codonfm-finetune/SKILL.md`)
- **MEDIUM** SECURITY/Unknown (LP3): MCP Least Privilege: The skill declares no explicit tool scope (no 'permissions' or 'allowed-tools' field in metadata), yet the skill content (`SKILL.md:1`)
- **MEDIUM** SECURITY/Autonomous Decision Making (EA2): Excessive Agency: without checking (`SKILL.md:162`)
- 11 additional finding(s) are available in the full evaluation artifacts.

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
