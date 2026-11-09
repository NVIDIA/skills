# Skill Benchmark: fastgen-validate

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `fastgen-validate`
- Validation status: `passed`
- Evaluation date: 2026-11-06
- Evaluator version: `1.6.8`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 4 evaluation tasks (4 positive)
- Dataset digest: `sha256:3977b1dfe3ac9c57751dc93a1bacbfab17b31ff4c8a10f83b4428bf0237cbf55` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| fastgen-validate | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| fastgen-validate | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| fastgen-validate | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| fastgen-validate | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| fastgen-validate | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `fastgen-validate`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 88.6% — baseline ran, but no comparable score was available; uplift unavailable | Not available |
| Security | 100.0% → 75.0% (-25.0 points) | Not available |
| Correctness | 100.0% → 100.0% (±0.0 points) | Not available |
| Discoverability | 81.3% — baseline ran, but no comparable score was available; uplift unavailable | Not available |
| Effectiveness | 97.5% → 100.0% (+2.5 points) | Not available |
| Efficiency | 86.8% — baseline ran, but no comparable score was available; uplift unavailable | Not available |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 92.6% → 92.8% (+0.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 885,441 | 1,027,887 | -142,446 | -13.86% | skill 4/4; base 4/4 |
| claude-code | adjacent-sample-generation | 149,000 | 149,451 | -451 | -0.30% | skill 1/1; base 1/1 |
| claude-code | contextual-claims-audit | 162,843 | 252,154 | -89,311 | -35.42% | skill 1/1; base 1/1 |
| claude-code | explicit-preserve-acceptance | 417,589 | 306,903 | +110,686 | +36.07% | skill 1/1; base 1/1 |
| claude-code | implicit-uncontrolled-parity | 156,009 | 319,379 | -163,370 | -51.15% | skill 1/1; base 1/1 |
| codex | All cases | 40,792 | 13,513 | N/A | N/A | skill 3/4; base 1/4; With-skill token coverage is incomplete (3/4 attempts). Some token counters are missing. Baseline token coverage is incomplete (1/4 attempts). Some token counters are missing. |
| codex | adjacent-sample-generation | 13,552 | 13,513 | +39 | +0.29% | skill 1/1; base 1/1 |
| codex | contextual-claims-audit | 13,667 | N/A | N/A | N/A | skill 1/1; base 0/1; Baseline token coverage is incomplete (0/1 attempts). Some token counters are missing. |
| codex | explicit-preserve-acceptance | N/A | N/A | N/A | N/A | skill 0/1; base 0/1; With-skill token coverage is incomplete (0/1 attempts). Some token counters are missing. Baseline token coverage is incomplete (0/1 attempts). Some token counters are missing. |
| codex | implicit-uncontrolled-parity | 13,573 | N/A | N/A | N/A | skill 1/1; base 0/1; Baseline token coverage is incomplete (0/1 attempts). Some token counters are missing. |
| ALL AGENTS | Dataset aggregate | 926,233 | 1,041,400 | N/A | N/A | skill 7/8; base 5/8; With-skill token coverage is incomplete (7/8 attempts). Some token counters are missing. Baseline token coverage is incomplete (5/8 attempts). Some token counters are missing. |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 13 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/fastgen-validate/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/fastgen-validate/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/fastgen-validate/SKILL.md`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/fastgen-runtime/SKILL.md (`SKILL.md:13`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/fastgen-inference/SKILL.md (`SKILL.md:14`)
- 8 additional finding(s) are available in the full evaluation artifacts.

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
