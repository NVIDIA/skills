# Skill Benchmark: tao-train-dino

> ❌ **Overall verdict: FAIL — Publication blocked**

The skill should be reviewed before publication. Address the blocking findings below, then rerun Skill Evaluator.

## Evaluation Metadata

- Skill: `tao-train-dino`
- Evaluation date: 2026-09-17
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 1 evaluation tasks (1 positive)
- Dataset digest: `sha256:a0a4e58efb79400761614702ba482a9d245b9f3b5fd462ee547295fa15ec910e` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 99.5% — baseline ran, but no comparable score was available; uplift unavailable | 94.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 0.0% → 100.0% (+100.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 5.6% → 100.0% (+94.4 points) | 36.7% → 78.3% (+41.6 points) |
| Efficiency | 97.4% — baseline ran, but no comparable score was available; uplift unavailable | 98.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 68,519 | 1,160,764 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | tao-train-dino-basic | 68,519 | 1,160,764 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 30,715 | 14,031 | +16,684 | +118.91% | skill 1/1; base 1/1 |
| codex | tao-train-dino-basic | 30,715 | 14,031 | +16,684 | +118.91% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 99,234 | 1,174,795 | N/A | N/A | skill 2/2; base 4/4 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **FAILED** | 11 validator(s); 31 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 3 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 1 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/dino-data-specs.md:
  "## Train Action Policy" in SKILL.md (lines 30-35)
  vs "## Train Action Policy" in references/dino-data-specs.md (lines 50-55) (`SKILL.md:30`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/dino-data-specs.md:
  "## Dataclass Schemas" in SKILL.md (lines 26-29)
  vs "## Dataclass Schemas" in references/dino-data-specs.md (lines 46-49) (`SKILL.md:26`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/dino-data-specs.md and references/tao-deploy-dino.md:
  "# DINO" in SKILL.md (lines 1-8)
  vs "## When To Use" in SKILL.md (lines 9-17)
  vs "# DINO" in references/dino-data-specs.md (lines 35-45)
  vs "# At runtime the SDK extracts it and points DINO at the extracted "images" folder." in references/dino-data-specs.md (lines 236-237)
  vs "# DINO Deploy" in references/tao-deploy-dino.md (lines 1-10)
  vs "### Generate TensorRT Engine" in references/tao-deploy-dino.md (lines 20-30)
  vs "### Evaluate TensorRT Engine" in references/tao-deploy-dino.md (lines 31-42)
  vs "### TensorRT Inference" in references/tao-deploy-dino.md (lines 43-61)
  vs "## Deploy Workflow" in references/tao-deploy-dino.md (lines 62-73)
  vs "## Required Inputs" in references/tao-deploy-dino.md (lines 74-93) (`SKILL.md:1`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/models/tao-train-dino/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'tags' is ignored; use 'metadata.tags' (`skills/models/tao-train-dino/SKILL.md`)
- 29 additional finding(s) are available in the full evaluation artifacts.

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
