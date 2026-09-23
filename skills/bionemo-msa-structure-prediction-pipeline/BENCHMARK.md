# Skill Benchmark: msa-structure-prediction-pipeline

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `msa-structure-prediction-pipeline`
- Evaluation date: 2026-09-19
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (4 positive)
- Dataset digest: `sha256:69feec5f4715fb1d6f8488fa2a2b2948f573b98d6347fd0538cf9f6a9d65ac8b` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 65.5% — baseline ran, but no comparable score was available; uplift unavailable | 83.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 60.0% → 50.0% (-10.0 points) | 37.5% → 100.0% (+62.5 points) |
| Correctness | 34.0% → 83.3% (+49.3 points) | 72.5% → 85.0% (+12.5 points) |
| Discoverability | 99.2% — baseline ran, but no comparable score was available; uplift unavailable | 86.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 11.6% → 26.1% (+14.5 points) | 40.2% → 58.3% (+18.1 points) |
| Efficiency | 68.7% — baseline ran, but no comparable score was available; uplift unavailable | 86.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,525,021 | 4,412,102 | N/A | N/A | skill 6/6; base 10/10 |
| claude-code | eval-1-basic-structure-prediction | 569,669 | 1,697,001 | N/A | N/A | skill 2/2; base 3/3 |
| claude-code | eval-2-high-depth-msa-retrieval | 349,168 | 457,656 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | eval-3-multi-database-msa | 372,967 | 670,230 | N/A | N/A | skill 2/2; base 1/1 |
| claude-code | eval-4-full-hosted-pipeline-with-scores | 233,217 | 1,587,215 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 608,000 | 3,848,751 | N/A | N/A | skill 4/4; base 8/8 |
| codex | eval-1-basic-structure-prediction | 117,699 | 906,297 | N/A | N/A | skill 1/1; base 3/3 |
| codex | eval-2-high-depth-msa-retrieval | 174,594 | 1,891,798 | N/A | N/A | skill 1/1; base 3/3 |
| codex | eval-3-multi-database-msa | 149,019 | 636,448 | -487,429 | -76.59% | skill 1/1; base 1/1 |
| codex | eval-4-full-hosted-pipeline-with-scores | 166,688 | 414,208 | -247,520 | -59.76% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,133,021 | 8,260,853 | N/A | N/A | skill 10/10; base 18/18 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 17 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/bionemo-agent-toolkit/skills/msa-structure-prediction-pipeline/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/bionemo-agent-toolkit/skills/msa-structure-prediction-pipeline/SKILL.md`)
- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/bionemo-agent-toolkit/skills/msa-structure-prediction-pipeline/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/bionemo-agent-toolkit/skills/msa-structure-prediction-pipeline`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/bionemo-agent-toolkit/skills/msa-structure-prediction-pipeline/SKILL.md`)
- 12 additional finding(s) are available in the full evaluation artifacts.

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
