# Skill Benchmark: nvalchemi-toolkit

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvalchemi-toolkit`
- Validation status: `passed`
- Evaluation date: 2026-11-14
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
| Overall | 91.1% — baseline ran, but no comparable score was available; uplift unavailable | 87.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 91.1% → 97.8% (+6.7 points) | 91.1% → 93.3% (+2.2 points) |
| Discoverability | 85.0% — baseline ran, but no comparable score was available; uplift unavailable | 85.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 83.2% → 89.2% (+6.0 points) | 75.3% → 83.4% (+8.1 points) |
| Efficiency | 83.7% — baseline ran, but no comparable score was available; uplift unavailable | 75.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 82.1% → 85.8% (+3.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 82.4% → 84.5% (+2.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,838,315 | 6,775,971 | -2,937,656 | -43.35% | skill 9/9; base 9/9 |
| claude-code | comp-01 | 719,895 | 2,097,205 | -1,377,310 | -65.67% | skill 1/1; base 1/1 |
| claude-code | imp-01 | 634,990 | 1,120,275 | -485,285 | -43.32% | skill 1/1; base 1/1 |
| claude-code | imp-02 | 940,346 | 1,195,029 | -254,683 | -21.31% | skill 1/1; base 1/1 |
| claude-code | model-01 | 551,397 | 731,236 | -179,839 | -24.59% | skill 1/1; base 1/1 |
| claude-code | neg-01 | 147,030 | 145,443 | +1,587 | +1.09% | skill 1/1; base 1/1 |
| claude-code | neg-03 | 268,521 | 308,685 | -40,164 | -13.01% | skill 1/1; base 1/1 |
| claude-code | scope-01 | 206,794 | 274,182 | -67,388 | -24.58% | skill 1/1; base 1/1 |
| claude-code | scope-02 | 295,471 | 793,948 | -498,477 | -62.78% | skill 1/1; base 1/1 |
| claude-code | scope-03 | 73,871 | 109,968 | -36,097 | -32.83% | skill 1/1; base 1/1 |
| codex | All cases | 6,173,076 | 8,348,524 | -2,175,448 | -26.06% | skill 9/9; base 9/9 |
| codex | comp-01 | 1,156,406 | 1,284,770 | -128,364 | -9.99% | skill 1/1; base 1/1 |
| codex | imp-01 | 1,271,564 | 1,216,485 | +55,079 | +4.53% | skill 1/1; base 1/1 |
| codex | imp-02 | 2,752,355 | 1,949,116 | +803,239 | +41.21% | skill 1/1; base 1/1 |
| codex | model-01 | 256,347 | 574,155 | -317,808 | -55.35% | skill 1/1; base 1/1 |
| codex | neg-01 | 74,458 | 73,506 | +952 | +1.30% | skill 1/1; base 1/1 |
| codex | neg-03 | 469,973 | 595,263 | -125,290 | -21.05% | skill 1/1; base 1/1 |
| codex | scope-01 | 49,178 | 45,303 | +3,875 | +8.55% | skill 1/1; base 1/1 |
| codex | scope-02 | 74,957 | 2,567,253 | -2,492,296 | -97.08% | skill 1/1; base 1/1 |
| codex | scope-03 | 67,838 | 42,673 | +25,165 | +58.97% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 10,011,391 | 15,124,495 | -5,113,104 | -33.81% | skill 18/18; base 18/18 |

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
- **MEDIUM** SECURITY/Unsafe Defaults (TM3): Tool Misuse: skip_validation (`references/data.md:218`)
- **MEDIUM** SECURITY/Unsafe Defaults (TM3): Tool Misuse: skip_validation (`references/data.md:86`)
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
