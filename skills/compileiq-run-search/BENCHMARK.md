# Skill Benchmark: compileiq-run-search

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `compileiq-run-search`
- Validation status: `passed`
- Evaluation date: 2026-11-14
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 8 evaluation tasks (6 positive, 2 negative)
- Dataset digest: `sha256:701c577c265244c03db5998862ab7d9963e6ca9f9d889907a5f39465fb99e61b` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| compileiq-run-search | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-run-search | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-run-search | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-run-search | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-run-search | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `compileiq-run-search`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 86.6% — baseline ran, but no comparable score was available; uplift unavailable | 88.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 93.8% → 100.0% (+6.2 points) |
| Correctness | 62.5% → 90.0% (+27.5 points) | 85.0% → 97.5% (+12.5 points) |
| Discoverability | 78.3% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 38.2% → 78.6% (+40.4 points) | 68.0% → 85.2% (+17.2 points) |
| Efficiency | 85.8% — baseline ran, but no comparable score was available; uplift unavailable | 65.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.8% → 97.0% (-0.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 85.3% → 88.4% (+3.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 973,882 | 1,088,143 | -114,261 | -10.50% | skill 8/8; base 8/8 |
| claude-code | compileiq-run-search-clock-lock-no-root | 106,415 | 181,097 | -74,682 | -41.24% | skill 1/1; base 1/1 |
| claude-code | compileiq-run-search-config-sizing | 150,619 | 181,699 | -31,080 | -17.11% | skill 1/1; base 1/1 |
| claude-code | compileiq-run-search-hang-isolated-worker | 187,325 | 181,622 | +5,703 | +3.14% | skill 1/1; base 1/1 |
| claude-code | compileiq-run-search-mlflow-tracker | 113,689 | 108,126 | +5,563 | +5.14% | skill 1/1; base 1/1 |
| claude-code | compileiq-run-search-negative-largest-files | 70,713 | 70,522 | +191 | +0.27% | skill 1/1; base 1/1 |
| claude-code | compileiq-run-search-negative-triton-autotune | 37,670 | 37,563 | +107 | +0.28% | skill 1/1; base 1/1 |
| claude-code | compileiq-run-search-ray-warnings | 192,862 | 181,411 | +11,451 | +6.31% | skill 1/1; base 1/1 |
| claude-code | compileiq-run-search-review-overnight-script | 114,589 | 146,103 | -31,514 | -21.57% | skill 1/1; base 1/1 |
| codex | All cases | 324,488 | 478,040 | -153,552 | -32.12% | skill 8/8; base 8/8 |
| codex | compileiq-run-search-clock-lock-no-root | 45,770 | 54,530 | -8,760 | -16.06% | skill 1/1; base 1/1 |
| codex | compileiq-run-search-config-sizing | 38,114 | 125,923 | -87,809 | -69.73% | skill 1/1; base 1/1 |
| codex | compileiq-run-search-hang-isolated-worker | 42,540 | 47,848 | -5,308 | -11.09% | skill 1/1; base 1/1 |
| codex | compileiq-run-search-mlflow-tracker | 37,635 | 42,554 | -4,919 | -11.56% | skill 1/1; base 1/1 |
| codex | compileiq-run-search-negative-largest-files | 13,471 | 13,302 | +169 | +1.27% | skill 1/1; base 1/1 |
| codex | compileiq-run-search-negative-triton-autotune | 27,182 | 30,798 | -3,616 | -11.74% | skill 1/1; base 1/1 |
| codex | compileiq-run-search-ray-warnings | 47,091 | 53,256 | -6,165 | -11.58% | skill 1/1; base 1/1 |
| codex | compileiq-run-search-review-overnight-script | 72,685 | 109,829 | -37,144 | -33.82% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,298,370 | 1,566,183 | -267,813 | -17.10% | skill 16/16; base 16/16 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 16 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/compileiq-run-search/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/compileiq-run-search/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/compileiq-run-search/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/compileiq-run-search/SKILL.md`)
- **MEDIUM** SECURITY/Sudo/Root Execution (PE2): Privilege Escalation: needs root privileges (`SKILL.md:178`)
- 11 additional finding(s) are available in the full evaluation artifacts.

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
