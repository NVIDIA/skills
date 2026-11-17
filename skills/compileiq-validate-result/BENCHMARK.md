# Skill Benchmark: compileiq-validate-result

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `compileiq-validate-result`
- Validation status: `passed`
- Evaluation date: 2026-11-14
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 8 evaluation tasks (6 positive, 2 negative)
- Dataset digest: `sha256:6901114747ad65942d6998dd3d4f88cf58994f875bdd3f21d7c6d6f05b5fde64` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| compileiq-validate-result | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-validate-result | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-validate-result | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-validate-result | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-validate-result | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `compileiq-validate-result`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 83.6% — baseline ran, but no comparable score was available; uplift unavailable | 91.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 93.8% → 100.0% (+6.2 points) |
| Correctness | 60.0% → 85.0% (+25.0 points) | 62.5% → 95.0% (+32.5 points) |
| Discoverability | 59.7% — baseline ran, but no comparable score was available; uplift unavailable | 87.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 51.6% → 78.1% (+26.5 points) | 55.0% → 88.1% (+33.1 points) |
| Efficiency | 95.4% — baseline ran, but no comparable score was available; uplift unavailable | 85.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.2% → 96.4% (-1.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 86.7% → 89.8% (+3.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,127,741 | 948,019 | +179,722 | +18.96% | skill 8/8; base 8/8 |
| claude-code | compileiq-validate-result-cli-dry-run | 247,261 | 222,443 | +24,818 | +11.16% | skill 1/1; base 1/1 |
| claude-code | compileiq-validate-result-genuine-win | 108,508 | 145,560 | -37,052 | -25.45% | skill 1/1; base 1/1 |
| claude-code | compileiq-validate-result-higher-variance | 247,081 | 183,501 | +63,580 | +34.65% | skill 1/1; base 1/1 |
| claude-code | compileiq-validate-result-lucky-min | 108,882 | 108,792 | +90 | +0.08% | skill 1/1; base 1/1 |
| claude-code | compileiq-validate-result-negative-generic-timing | 36,278 | 35,989 | +289 | +0.80% | skill 1/1; base 1/1 |
| claude-code | compileiq-validate-result-negative-largest-files | 70,150 | 70,755 | -605 | -0.86% | skill 1/1; base 1/1 |
| claude-code | compileiq-validate-result-reuse-and-pipe | 162,454 | 36,465 | +125,989 | +345.51% | skill 1/1; base 1/1 |
| claude-code | compileiq-validate-result-topk-csv | 147,127 | 144,514 | +2,613 | +1.81% | skill 1/1; base 1/1 |
| codex | All cases | 623,059 | 735,597 | -112,538 | -15.30% | skill 8/8; base 8/8 |
| codex | compileiq-validate-result-cli-dry-run | 131,955 | 98,723 | +33,232 | +33.66% | skill 1/1; base 1/1 |
| codex | compileiq-validate-result-genuine-win | 96,713 | 146,968 | -50,255 | -34.19% | skill 1/1; base 1/1 |
| codex | compileiq-validate-result-higher-variance | 142,128 | 199,896 | -57,768 | -28.90% | skill 1/1; base 1/1 |
| codex | compileiq-validate-result-lucky-min | 65,225 | 77,695 | -12,470 | -16.05% | skill 1/1; base 1/1 |
| codex | compileiq-validate-result-negative-generic-timing | 23,004 | 27,726 | -4,722 | -17.03% | skill 1/1; base 1/1 |
| codex | compileiq-validate-result-negative-largest-files | 13,546 | 13,342 | +204 | +1.53% | skill 1/1; base 1/1 |
| codex | compileiq-validate-result-reuse-and-pipe | 68,146 | 85,989 | -17,843 | -20.75% | skill 1/1; base 1/1 |
| codex | compileiq-validate-result-topk-csv | 82,342 | 85,258 | -2,916 | -3.42% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,750,800 | 1,683,616 | +67,184 | +3.99% | skill 16/16; base 16/16 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 23 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and scripts/welch_validate.py:
  "### 4. Statistical gate — the ship rule" in SKILL.md (lines 70-100)
  vs "(module docstring)" in scripts/welch_validate.py (lines 1-12)
  vs "validate_speedup()" in scripts/welch_validate.py (lines 103-144) (`SKILL.md:70`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/compileiq-validate-result/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/compileiq-validate-result/SKILL.md`)
- **MEDIUM** QUALITY/quality_discoverability: Description uses first/second person (`skills/compileiq-validate-result/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/compileiq-validate-result/SKILL.md`)
- 19 additional finding(s) are available in the full evaluation artifacts.

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
