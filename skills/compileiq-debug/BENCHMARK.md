# Skill Benchmark: compileiq-debug

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `compileiq-debug`
- Validation status: `passed`
- Evaluation date: 2026-11-14
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 8 evaluation tasks (6 positive, 2 negative)
- Dataset digest: `sha256:8ed6852b90ceee0271a5e7784520510f89b65ecf1ed06d44bab90070712127ae` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| compileiq-debug | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-debug | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-debug | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-debug | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-debug | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `compileiq-debug`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | Not available | 89.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 100.0% → 100.0% (±0.0 points) |
| Correctness | Not available | 92.5% → 100.0% (+7.5 points) |
| Discoverability | Not available | 94.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 76.6% → 85.4% (+8.8 points) |
| Efficiency | Not available | 68.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 88.0% → 89.0% (+1.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 719,945 | 1,026,380 | -306,435 | -29.86% | skill 8/8; base 8/8 |
| claude-code | compileiq-debug-csv-rising-invalid | 146,468 | 146,017 | +451 | +0.31% | skill 1/1; base 1/1 |
| claude-code | compileiq-debug-first-eval-hang | 147,939 | 257,242 | -109,303 | -42.49% | skill 1/1; base 1/1 |
| claude-code | compileiq-debug-fromhex | 0 | 70,990 | -70,990 | -100.00% | skill 1/1; base 1/1 |
| claude-code | compileiq-debug-high-cv | 36,478 | 36,382 | +96 | +0.26% | skill 1/1; base 1/1 |
| claude-code | compileiq-debug-invalid-score-tuple | 112,542 | 259,542 | -147,000 | -56.64% | skill 1/1; base 1/1 |
| claude-code | compileiq-debug-negative-nvcc-install | 35,980 | 35,887 | +93 | +0.26% | skill 1/1; base 1/1 |
| claude-code | compileiq-debug-negative-pytorch-nan | 36,412 | 36,471 | -59 | -0.16% | skill 1/1; base 1/1 |
| claude-code | compileiq-debug-same-score-triton | 204,126 | 183,849 | +20,277 | +11.03% | skill 1/1; base 1/1 |
| codex | All cases | 348,579 | 335,253 | +13,326 | +3.97% | skill 8/8; base 8/8 |
| codex | compileiq-debug-csv-rising-invalid | 90,886 | 120,988 | -30,102 | -24.88% | skill 1/1; base 1/1 |
| codex | compileiq-debug-first-eval-hang | 38,906 | 27,881 | +11,025 | +39.54% | skill 1/1; base 1/1 |
| codex | compileiq-debug-fromhex | 38,650 | 50,743 | -12,093 | -23.83% | skill 1/1; base 1/1 |
| codex | compileiq-debug-high-cv | 36,069 | 27,815 | +8,254 | +29.67% | skill 1/1; base 1/1 |
| codex | compileiq-debug-invalid-score-tuple | 34,270 | 21,709 | +12,561 | +57.86% | skill 1/1; base 1/1 |
| codex | compileiq-debug-negative-nvcc-install | 26,444 | 26,470 | -26 | -0.10% | skill 1/1; base 1/1 |
| codex | compileiq-debug-negative-pytorch-nan | 23,836 | 23,196 | +640 | +2.76% | skill 1/1; base 1/1 |
| codex | compileiq-debug-same-score-triton | 59,518 | 36,451 | +23,067 | +63.28% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,068,524 | 1,361,633 | -293,109 | -21.53% | skill 16/16; base 16/16 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 18 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/compileiq-debug/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/compileiq-debug/SKILL.md`)
- **MEDIUM** QUALITY/quality_discoverability: Description contains vague words (`skills/compileiq-debug/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/compileiq-debug/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/compileiq-debug/SKILL.md`)
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
