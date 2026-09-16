# Skill Benchmark: openfold3-nim

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `openfold3-nim`
- Evaluation date: 2026-09-30
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (4 positive)
- Dataset digest: `sha256:405d82d91432042dcd7576e872a7acae051a4a4fef41962393f6e84e901b1513` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 91.2% — baseline ran, but no comparable score was available; uplift unavailable | 83.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 50.0% → 87.5% (+37.5 points) | 30.0% → 75.0% (+45.0 points) |
| Correctness | 85.0% → 100.0% (+15.0 points) | 92.0% → 95.0% (+3.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 91.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 71.0% → 88.4% (+17.4 points) | 61.2% → 77.0% (+15.8 points) |
| Efficiency | 79.8% — baseline ran, but no comparable score was available; uplift unavailable | 79.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 677,568 | 3,030,703 | -2,353,135 | -77.64% | skill 4/4; base 4/4 |
| claude-code | 1 | 199,955 | 580,721 | -380,766 | -65.57% | skill 1/1; base 1/1 |
| claude-code | 2 | 164,711 | 2,055,944 | -1,891,233 | -91.99% | skill 1/1; base 1/1 |
| claude-code | 3 | 173,566 | 186,230 | -12,664 | -6.80% | skill 1/1; base 1/1 |
| claude-code | 4 | 139,336 | 207,808 | -68,472 | -32.95% | skill 1/1; base 1/1 |
| codex | All cases | 457,858 | 1,176,860 | N/A | N/A | skill 4/4; base 5/5 |
| codex | 1 | 99,872 | 222,120 | -122,248 | -55.04% | skill 1/1; base 1/1 |
| codex | 2 | 99,645 | 254,679 | -155,034 | -60.87% | skill 1/1; base 1/1 |
| codex | 3 | 190,961 | 675,195 | N/A | N/A | skill 1/1; base 2/2 |
| codex | 4 | 67,380 | 24,866 | +42,514 | +170.97% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,135,426 | 4,207,563 | N/A | N/A | skill 8/8; base 9/9 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 25 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/api.md:
  "## Local Docker" in SKILL.md (lines 41-86)
  vs "## Docker Reference" in references/api.md (lines 117-160) (`SKILL.md:41`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/bionemo-agent-toolkit/skills/openfold3-nim/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/bionemo-agent-toolkit/skills/openfold3-nim/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in parameters.md (`skills/bionemo-agent-toolkit/skills/openfold3-nim/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/bionemo-agent-toolkit/skills/openfold3-nim`)
- 21 additional finding(s) are available in the full evaluation artifacts.

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
