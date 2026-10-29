# Skill Benchmark: nemotron-voice-agent-builder

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemotron-voice-agent-builder`
- Validation status: `passed`
- Evaluation date: 2026-10-29
- Evaluator version: `1.6.4`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 5 evaluation tasks (4 positive, 1 negative)
- Dataset digest: `sha256:820305d412563ab26a2312566e2cd07ced079c651a147528990951cee89bcebc` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nemotron-voice-agent-builder | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |
| nemotron-voice-agent-builder | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |
| nemotron-voice-agent-builder | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |
| nemotron-voice-agent-builder | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |
| nemotron-voice-agent-builder | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.2` | nondeterministic / provider_defaults |

- Controls for `nemotron-voice-agent-builder`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 80.6% — baseline ran, but no comparable score was available; uplift unavailable | 79.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 40.0% → 80.0% (+40.0 points) | 50.0% → 90.0% (+40.0 points) |
| Correctness | 52.0% → 92.0% (+40.0 points) | 64.0% → 80.0% (+16.0 points) |
| Discoverability | 95.0% — baseline ran, but no comparable score was available; uplift unavailable | 93.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 34.0% → 46.0% (+12.0 points) | 28.0% → 50.0% (+22.0 points) |
| Efficiency | 89.9% — baseline ran, but no comparable score was available; uplift unavailable | 81.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 79.4% → 86.0% (+6.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 65.9% → 83.4% (+17.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,820,620 | 11,066,068 | -9,245,448 | -83.55% | skill 5/5; base 5/5 |
| claude-code | nemotron-voice-agent-builder-contextual-high-concurrency-workstation | 313,230 | 149,670 | +163,560 | +109.28% | skill 1/1; base 1/1 |
| claude-code | nemotron-voice-agent-builder-explicit-dgx-spark | 418,046 | 109,476 | +308,570 | +281.86% | skill 1/1; base 1/1 |
| claude-code | nemotron-voice-agent-builder-implicit-low-concurrency-workstation | 326,037 | 8,803,520 | -8,477,483 | -96.30% | skill 1/1; base 1/1 |
| claude-code | nemotron-voice-agent-builder-livekit-omni-conflict | 265,235 | 1,510,618 | -1,245,383 | -82.44% | skill 1/1; base 1/1 |
| claude-code | nemotron-voice-agent-builder-negative-text-rag | 498,072 | 492,784 | +5,288 | +1.07% | skill 1/1; base 1/1 |
| codex | All cases | 2,407,700 | 12,620,044 | -10,212,344 | -80.92% | skill 5/5; base 5/5 |
| codex | nemotron-voice-agent-builder-contextual-high-concurrency-workstation | 98,307 | 47,826 | +50,481 | +105.55% | skill 1/1; base 1/1 |
| codex | nemotron-voice-agent-builder-explicit-dgx-spark | 84,192 | 62,537 | +21,655 | +34.63% | skill 1/1; base 1/1 |
| codex | nemotron-voice-agent-builder-implicit-low-concurrency-workstation | 113,170 | 3,941,008 | -3,827,838 | -97.13% | skill 1/1; base 1/1 |
| codex | nemotron-voice-agent-builder-livekit-omni-conflict | 49,098 | 2,108,713 | -2,059,615 | -97.67% | skill 1/1; base 1/1 |
| codex | nemotron-voice-agent-builder-negative-text-rag | 2,062,933 | 6,459,960 | -4,397,027 | -68.07% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 4,228,320 | 23,686,112 | -19,457,792 | -82.15% | skill 10/10; base 10/10 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 17 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **CRITICAL** CONTENT_DEDUP/llm_error: LLM analysis failed for a content cluster (`skills/nemotron-voice-agent-builder`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/nemotron-voice-agent-builder/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nemotron-voice-agent-builder/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nemotron-voice-agent-builder/SKILL.md`)
- **MEDIUM** SECURITY/Autonomous Decision Making (EA2): Excessive Agency: without checking (`references/domain/speech-customization.md:110`)
- 13 additional finding(s) are available in the full evaluation artifacts.

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
