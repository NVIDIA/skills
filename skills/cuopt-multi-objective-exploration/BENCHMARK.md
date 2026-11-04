# Skill Benchmark: cuopt-multi-objective-exploration

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cuopt-multi-objective-exploration`
- Evaluation date: 2026-10-23
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 8 evaluation tasks (6 positive, 2 negative)
- Dataset digest: `sha256:91da323e5d8f4a5c51ce0ff53037c3876cc0bb17fc8109347b415577c965ab55` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 89.1% — baseline ran, but no comparable score was available; uplift unavailable | 82.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 67.5% → 100.0% (+32.5 points) | 90.0% → 85.0% (-5.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 46.9% → 48.5% (+1.6 points) | 65.9% → 62.9% (-3.0 points) |
| Efficiency | 97.1% — baseline ran, but no comparable score was available; uplift unavailable | 69.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 909,008 | 827,867 | +81,141 | +9.80% | skill 8/8; base 8/8 |
| claude-code | multiobj-explore-eval-001-supplier-interpretation | 69,273 | 33,423 | +35,850 | +107.26% | skill 1/1; base 1/1 |
| claude-code | multiobj-explore-eval-002-supplier-exploration | 142,836 | 94,134 | +48,702 | +51.74% | skill 1/1; base 1/1 |
| claude-code | multiobj-explore-eval-003-single-objective-decoy | 216,656 | 120,149 | +96,507 | +80.32% | skill 1/1; base 1/1 |
| claude-code | multiobj-explore-eval-004-dual-exchange-rate | 71,666 | 240,462 | -168,796 | -70.20% | skill 1/1; base 1/1 |
| claude-code | multiobj-explore-eval-005-latent-objective | 103,195 | 124,622 | -21,427 | -17.19% | skill 1/1; base 1/1 |
| claude-code | multiobj-explore-eval-006-gap-detection-completion | 102,300 | 61,703 | +40,597 | +65.79% | skill 1/1; base 1/1 |
| claude-code | multiobj-explore-eval-007-provenance-reporting | 65,902 | 32,780 | +33,122 | +101.04% | skill 1/1; base 1/1 |
| claude-code | multiobj-explore-eval-008-single-objective-lp-decoy | 137,180 | 120,594 | +16,586 | +13.75% | skill 1/1; base 1/1 |
| codex | All cases | 336,884 | 187,547 | +149,337 | +79.63% | skill 8/8; base 8/8 |
| codex | multiobj-explore-eval-001-supplier-interpretation | 30,259 | 13,973 | +16,286 | +116.55% | skill 1/1; base 1/1 |
| codex | multiobj-explore-eval-002-supplier-exploration | 57,442 | 31,053 | +26,389 | +84.98% | skill 1/1; base 1/1 |
| codex | multiobj-explore-eval-003-single-objective-decoy | 41,153 | 27,213 | +13,940 | +51.23% | skill 1/1; base 1/1 |
| codex | multiobj-explore-eval-004-dual-exchange-rate | 51,340 | 39,215 | +12,125 | +30.92% | skill 1/1; base 1/1 |
| codex | multiobj-explore-eval-005-latent-objective | 38,843 | 22,439 | +16,404 | +73.10% | skill 1/1; base 1/1 |
| codex | multiobj-explore-eval-006-gap-detection-completion | 42,720 | 17,960 | +24,760 | +137.86% | skill 1/1; base 1/1 |
| codex | multiobj-explore-eval-007-provenance-reporting | 34,084 | 22,291 | +11,793 | +52.90% | skill 1/1; base 1/1 |
| codex | multiobj-explore-eval-008-single-objective-lp-decoy | 41,043 | 13,403 | +27,640 | +206.22% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,245,892 | 1,015,414 | +230,478 | +22.70% | skill 16/16; base 16/16 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/cuopt-multi-objective-exploration/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/cuopt-multi-objective-exploration/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/cuopt-multi-objective-exploration/SKILL.md`)
- **MEDIUM** SECURITY/Unknown (AE4): analysis-evasion: Suspicious Unicode normalization or mixed-script content (`SKILL.md:1`)
- **LOW** QUALITY/quality_discoverability: Description doesn't mention WHEN to use this skill (`skills/cuopt-multi-objective-exploration/SKILL.md`)
- 7 additional finding(s) are available in the full evaluation artifacts.

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
