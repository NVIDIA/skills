# Skill Benchmark: cudnn-install

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cudnn-install`
- Validation status: `passed`
- Evaluation date: 2026-11-20
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 10 evaluation tasks (8 positive, 2 negative)
- Dataset digest: `sha256:a66ebccfdd393ca0b919b7f133ba1d99a3ffb1dbdf56e50c5127129b615cbe58` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| cudnn-install | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| cudnn-install | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| cudnn-install | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| cudnn-install | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| cudnn-install | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `cudnn-install`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 93.7% — baseline ran, but no comparable score was available; uplift unavailable | 85.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 98.0% → 100.0% (+2.0 points) | 100.0% → 98.0% (-2.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 87.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 77.9% → 72.1% (-5.8 points) | 79.6% → 80.8% (+1.2 points) |
| Efficiency | 96.5% — baseline ran, but no comparable score was available; uplift unavailable | 59.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.1% → 96.2% (-1.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 86.4% → 86.9% (+0.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 776,939 | 587,455 | +189,484 | +32.26% | skill 10/10; base 10/10 |
| claude-code | cudnn-install-cpp-only-build | 161,971 | 109,418 | +52,553 | +48.03% | skill 1/1; base 1/1 |
| claude-code | cudnn-install-cpu-only-verification | 76,352 | 107,813 | -31,461 | -29.18% | skill 1/1; base 1/1 |
| claude-code | cudnn-install-cuda12-wheel-no-checkout | 78,282 | 37,272 | +41,010 | +110.03% | skill 1/1; base 1/1 |
| claude-code | cudnn-install-metadata-extras-and-groups | 78,376 | 74,193 | +4,183 | +5.64% | skill 1/1; base 1/1 |
| claude-code | cudnn-install-negative-executable-plan-cache | 38,195 | 37,923 | +272 | +0.72% | skill 1/1; base 1/1 |
| claude-code | cudnn-install-negative-python-list | 35,081 | 34,943 | +138 | +0.39% | skill 1/1; base 1/1 |
| claude-code | cudnn-install-ngc-no-pip-record | 76,143 | 36,227 | +39,916 | +110.18% | skill 1/1; base 1/1 |
| claude-code | cudnn-install-pytorch-version-override | 77,153 | 37,712 | +39,441 | +104.58% | skill 1/1; base 1/1 |
| claude-code | cudnn-install-reuse-framework-backend | 77,326 | 38,195 | +39,131 | +102.45% | skill 1/1; base 1/1 |
| claude-code | cudnn-install-upgrade-frontend-only | 78,060 | 73,759 | +4,301 | +5.83% | skill 1/1; base 1/1 |
| codex | All cases | 445,102 | 331,725 | +113,377 | +34.18% | skill 10/10; base 10/10 |
| codex | cudnn-install-cpp-only-build | 88,584 | 44,945 | +43,639 | +97.09% | skill 1/1; base 1/1 |
| codex | cudnn-install-cpu-only-verification | 33,697 | 25,946 | +7,751 | +29.87% | skill 1/1; base 1/1 |
| codex | cudnn-install-cuda12-wheel-no-checkout | 52,192 | 39,907 | +12,285 | +30.78% | skill 1/1; base 1/1 |
| codex | cudnn-install-metadata-extras-and-groups | 30,904 | 30,593 | +311 | +1.02% | skill 1/1; base 1/1 |
| codex | cudnn-install-negative-executable-plan-cache | 63,789 | 47,843 | +15,946 | +33.33% | skill 1/1; base 1/1 |
| codex | cudnn-install-negative-python-list | 13,264 | 13,194 | +70 | +0.53% | skill 1/1; base 1/1 |
| codex | cudnn-install-ngc-no-pip-record | 38,601 | 22,458 | +16,143 | +71.88% | skill 1/1; base 1/1 |
| codex | cudnn-install-pytorch-version-override | 47,108 | 36,008 | +11,100 | +30.83% | skill 1/1; base 1/1 |
| codex | cudnn-install-reuse-framework-backend | 38,470 | 30,818 | +7,652 | +24.83% | skill 1/1; base 1/1 |
| codex | cudnn-install-upgrade-frontend-only | 38,493 | 40,013 | -1,520 | -3.80% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,222,041 | 919,180 | +302,861 | +32.95% | skill 20/20; base 20/20 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 10 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 10 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/cudnn-install/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/cudnn-install/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/cudnn-install/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/cudnn-install/SKILL.md`)
- **MEDIUM** SCHEMA/author_missing: Author not specified in metadata (`skills/cudnn-install/SKILL.md`)
- 5 additional finding(s) are available in the full evaluation artifacts.

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
