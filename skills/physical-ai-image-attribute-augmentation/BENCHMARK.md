# Skill Benchmark: physical-ai-image-attribute-augmentation

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `physical-ai-image-attribute-augmentation`
- Evaluation date: 2026-09-02
- Evaluator version: `1.3.2`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 13 evaluation tasks (11 positive, 2 negative)
- Dataset digest: `sha256:99c88f504e8f80580bc8aeaaa4fa48a2cbf2e11790619156b347231167b6b96b` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Execution and Provenance

- Validation status: `passed`
- Report generation: `complete`
- Evaluator version: `1.3.2`
- Git commit: `82224ec75d08e41cc5b92f7bf935f69c0def4607`
- Content type: requested `auto`, detected `skill`
- Container image: `gitlab-master.nvidia.com:5005/nvcarps/ci-group/nvcarps-ci/skillevaluator-ci:sha-82224ec75d08e41cc5b92f7bf935f69c0def4607`
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
| Overall | 52% → 77% (+25 points) | 53% → 75% (+22 points) |
| Security | 85% → 100% (+15 points) | 81% → 96% (+15 points) |
| Correctness | 43% → 92% (+49 points) | 46% → 85% (+38 points) |
| Discoverability | 40% → 68% (+28 points) | 37% → 66% (+29 points) |
| Effectiveness | 46% → 75% (+30 points) | 48% → 66% (+18 points) |
| Efficiency | 45% → 48% (+4 points) | 53% → 62% (+9 points) |

**How to read this table:** baseline is the same task attempted without the target skill. Uplift is `skill score - baseline score`, shown in percentage points.

Example: `47% → 92% (+45 points)` means the skill-assisted run scored 92%, 45 percentage points above its 47% no-skill baseline.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 1 validator(s); 4 finding(s) |
| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 13 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/physical-ai-image-attribute-augmentation/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/physical-ai-image-attribute-augmentation/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/physical-ai-image-attribute-augmentation/SKILL.md`)
- **LOW** SCHEMA/author_format: Author must be of the form 'Name <email@host>' (`skills/physical-ai-image-attribute-augmentation/SKILL.md`)

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
