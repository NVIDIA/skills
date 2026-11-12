# Skill Benchmark: newton-usage

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `newton-usage`
- Validation status: `passed`
- Evaluation date: 2026-10-30
- Evaluator version: `1.6.4`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 5 evaluation tasks (3 positive, 2 negative)
- Dataset digest: `sha256:f186dac7d2c4e6f63962549401aecd2144324c46da0fee1334c49af5fc19b756` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| newton-usage | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |
| newton-usage | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |
| newton-usage | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |
| newton-usage | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |
| newton-usage | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |

- Controls for `newton-usage`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 97.7% — baseline ran, but no comparable score was available; uplift unavailable | 89.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 80.0% → 92.0% (+12.0 points) | 80.0% → 100.0% (+20.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 93.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 66.0% → 97.5% (+31.5 points) | 58.0% → 87.5% (+29.5 points) |
| Efficiency | 99.2% — baseline ran, but no comparable score was available; uplift unavailable | 68.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 93.1% → 98.4% (+5.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 90.2% → 90.7% (+0.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 396,425 | 286,834 | +109,591 | +38.21% | skill 5/5; base 5/5 |
| claude-code | newton-usage-context-version | 72,528 | 36,423 | +36,105 | +99.13% | skill 1/1; base 1/1 |
| claude-code | newton-usage-explicit-setup | 71,708 | 36,131 | +35,577 | +98.47% | skill 1/1; base 1/1 |
| claude-code | newton-usage-implicit-debug | 72,299 | 36,072 | +36,227 | +100.43% | skill 1/1; base 1/1 |
| claude-code | newton-usage-negative-contributor-ci | 144,772 | 143,202 | +1,570 | +1.10% | skill 1/1; base 1/1 |
| claude-code | newton-usage-negative-root-finding | 35,118 | 35,006 | +112 | +0.32% | skill 1/1; base 1/1 |
| codex | All cases | 211,526 | 164,764 | +46,762 | +28.38% | skill 5/5; base 5/5 |
| codex | newton-usage-context-version | 36,630 | 31,334 | +5,296 | +16.90% | skill 1/1; base 1/1 |
| codex | newton-usage-explicit-setup | 79,373 | 35,519 | +43,854 | +123.47% | skill 1/1; base 1/1 |
| codex | newton-usage-implicit-debug | 31,689 | 26,146 | +5,543 | +21.20% | skill 1/1; base 1/1 |
| codex | newton-usage-negative-contributor-ci | 50,513 | 58,514 | -8,001 | -13.67% | skill 1/1; base 1/1 |
| codex | newton-usage-negative-root-finding | 13,321 | 13,251 | +70 | +0.53% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 607,951 | 451,598 | +156,353 | +34.62% | skill 10/10; base 10/10 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 11 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/newton-usage/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/newton-usage/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/newton-usage/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/newton-usage/SKILL.md`)
- **MEDIUM** SCHEMA/author_missing: Author not specified in metadata (`skills/newton-usage/SKILL.md`)
- 6 additional finding(s) are available in the full evaluation artifacts.

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
