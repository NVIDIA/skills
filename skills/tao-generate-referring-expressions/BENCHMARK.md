# Skill Benchmark: tao-generate-referring-expressions

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `tao-generate-referring-expressions`
- Evaluation date: 2026-08-24
- Evaluator version: `1.3.2`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 1 evaluation tasks (1 positive)
- Dataset digest: `sha256:5d32143561ff4b116cf31ee3c4b782a64cf164ba5aad47c65d8f71564c34eb09` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `local`
- Tier 3 evidence: required for publication

Tasks ran on the trusted local host; local mode is not sandboxed.

## Execution and Provenance

- Validation status: `passed`
- Report generation: `complete`
- Evaluator version: `1.3.2`
- Git commit: `0117bc2e3e54da4244a656466526c5b1b5a559ea`
- Content type: requested `auto`, detected `skill`
- Container image: `gitlab-master.nvidia.com:5005/nvcarps/ci-group/nvcarps-ci/skillevaluator-ci:sha-0117bc2e3e54da4244a656466526c5b1b5a559ea`
- Container image digest: `not recorded`
- Tier 3: requested `true`, executed `true`, status `succeeded`

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
| Overall | 31% → 96% (+65 points) | 33% → 60% (+27 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 20% → 100% (+80 points) | 20% → 100% (+80 points) |
| Discoverability | 0% → 100% (+100 points) | 0% → 0% (±0 points) |
| Effectiveness | 33% → 95% (+62 points) | 43% → 100% (+57 points) |
| Efficiency | 0% → 83% (+83 points) | 0% → 0% (±0 points) |

**How to read this table:** baseline is the same task attempted without the target skill. Uplift is `skill score - baseline score`, shown in percentage points.

Example: `47% → 92% (+45 points)` means the skill-assisted run scored 92%, 45 percentage points above its 47% no-skill baseline.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 1 validator(s); 4 finding(s) |
| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 1 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'tags' is ignored; use 'metadata.tags' (`skills/data/tao-generate-referring-expressions/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/data/tao-generate-referring-expressions`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/data/tao-generate-referring-expressions/SKILL.md`)
- **LOW** SCHEMA/author_format: Author must be of the form 'Name <email@host>' (`skills/data/tao-generate-referring-expressions/SKILL.md`)

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
| Efficiency | Did it avoid wasted tool or skill usage? | `skill_efficiency` (100%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Token efficiency is a separate report-only signal. It does not change a dimension score or the overall verdict.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was found and executed.
- `skill_efficiency` (Efficiency): routing quality, workspace-aware skill reads, and productive tool use.
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
