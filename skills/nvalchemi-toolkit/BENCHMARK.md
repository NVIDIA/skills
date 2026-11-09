# Skill Benchmark: nvalchemi-toolkit

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvalchemi-toolkit`
- Validation status: `passed`
- Evaluation date: 2026-11-07
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 9 evaluation tasks (7 positive, 2 negative)
- Dataset digest: `sha256:c096747a831a786dceff6fcf5d46191b44347068431663c2a7e66efbf5593c7a` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nvalchemi-toolkit | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvalchemi-toolkit | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvalchemi-toolkit | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvalchemi-toolkit | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvalchemi-toolkit | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `nvalchemi-toolkit`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 93.1% — baseline ran, but no comparable score was available; uplift unavailable | 88.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 94.4% → 100.0% (+5.6 points) | 94.4% → 100.0% (+5.6 points) |
| Correctness | 88.9% → 95.6% (+6.7 points) | 82.2% → 97.8% (+15.6 points) |
| Discoverability | 95.0% — baseline ran, but no comparable score was available; uplift unavailable | 82.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 76.7% → 87.3% (+10.6 points) | 70.8% → 87.0% (+16.2 points) |
| Efficiency | 87.6% — baseline ran, but no comparable score was available; uplift unavailable | 76.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 80.9% → 85.3% (+4.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 81.3% → 84.1% (+2.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,767,299 | 8,028,455 | -4,261,156 | -53.08% | skill 9/9; base 9/9 |
| claude-code | comp-01 | 490,723 | 1,713,327 | -1,222,604 | -71.36% | skill 1/1; base 1/1 |
| claude-code | imp-01 | 659,070 | 1,273,754 | -614,684 | -48.26% | skill 1/1; base 1/1 |
| claude-code | imp-02 | 1,104,796 | 2,855,151 | -1,750,355 | -61.31% | skill 1/1; base 1/1 |
| claude-code | model-01 | 545,495 | 774,709 | -229,214 | -29.59% | skill 1/1; base 1/1 |
| claude-code | neg-01 | 148,013 | 182,912 | -34,899 | -19.08% | skill 1/1; base 1/1 |
| claude-code | neg-03 | 263,947 | 228,956 | +34,991 | +15.28% | skill 1/1; base 1/1 |
| claude-code | scope-01 | 195,314 | 192,612 | +2,702 | +1.40% | skill 1/1; base 1/1 |
| claude-code | scope-02 | 236,875 | 697,472 | -460,597 | -66.04% | skill 1/1; base 1/1 |
| claude-code | scope-03 | 123,066 | 109,562 | +13,504 | +12.33% | skill 1/1; base 1/1 |
| codex | All cases | 5,294,723 | 8,668,411 | -3,373,688 | -38.92% | skill 9/9; base 9/9 |
| codex | comp-01 | 1,075,998 | 827,111 | +248,887 | +30.09% | skill 1/1; base 1/1 |
| codex | imp-01 | 857,644 | 1,556,633 | -698,989 | -44.90% | skill 1/1; base 1/1 |
| codex | imp-02 | 2,304,597 | 1,957,890 | +346,707 | +17.71% | skill 1/1; base 1/1 |
| codex | model-01 | 328,780 | 657,059 | -328,279 | -49.96% | skill 1/1; base 1/1 |
| codex | neg-01 | 76,398 | 73,345 | +3,053 | +4.16% | skill 1/1; base 1/1 |
| codex | neg-03 | 454,227 | 414,425 | +39,802 | +9.60% | skill 1/1; base 1/1 |
| codex | scope-01 | 48,978 | 502,825 | -453,847 | -90.26% | skill 1/1; base 1/1 |
| codex | scope-02 | 79,241 | 2,635,925 | -2,556,684 | -96.99% | skill 1/1; base 1/1 |
| codex | scope-03 | 68,860 | 43,198 | +25,662 | +59.41% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 9,062,022 | 16,696,866 | -7,634,844 | -45.73% | skill 18/18; base 18/18 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 6 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 9 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nvalchemi-toolkit/SKILL.md`)
- **MEDIUM** SECURITY/Unsafe Defaults (TM3): Tool Misuse: skip_validation (`references/data.md:86`)
- **MEDIUM** SECURITY/Unsafe Defaults (TM3): Tool Misuse: skip_validation (`references/data.md:209`)
- **LOW** QUALITY/quality_discoverability: Description very long (703 chars, recommend 50-150) (`skills/nvalchemi-toolkit/SKILL.md`)
- **LOW** QUALITY/quality_reliability: Inputs are used but no dedicated Inputs section is documented (`skills/nvalchemi-toolkit/SKILL.md`)
- 1 additional finding(s) are available in the full evaluation artifacts.

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
