# Skill Benchmark: compileiq-booster-pack

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `compileiq-booster-pack`
- Validation status: `passed`
- Evaluation date: 2026-11-14
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 8 evaluation tasks (6 positive, 2 negative)
- Dataset digest: `sha256:999fc35cc7916a152827262764788e8b1570b506d4e999c118c75e504bd2337a` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| compileiq-booster-pack | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-booster-pack | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-booster-pack | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-booster-pack | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-booster-pack | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `compileiq-booster-pack`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 85.5% — baseline ran, but no comparable score was available; uplift unavailable | 88.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 55.0% → 95.0% (+40.0 points) | 80.0% → 97.5% (+17.5 points) |
| Discoverability | 60.8% — baseline ran, but no comparable score was available; uplift unavailable | 93.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 47.8% → 86.5% (+38.7 points) | 57.6% → 87.2% (+29.6 points) |
| Efficiency | 84.9% — baseline ran, but no comparable score was available; uplift unavailable | 65.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.5% → 96.3% (-1.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 86.7% → 88.3% (+1.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,077,981 | 819,462 | +258,519 | +31.55% | skill 8/8; base 8/8 |
| claude-code | compileiq-booster-pack-apply-one-acf-mock-bench | 230,272 | 158,823 | +71,449 | +44.99% | skill 1/1; base 1/1 |
| claude-code | compileiq-booster-pack-canary-o0-no-regress | 115,859 | 35,996 | +79,863 | +221.87% | skill 1/1; base 1/1 |
| claude-code | compileiq-booster-pack-download-and-prereqs | 118,118 | 110,193 | +7,925 | +7.19% | skill 1/1; base 1/1 |
| claude-code | compileiq-booster-pack-flashinfer-batch-decode | 165,391 | 150,747 | +14,644 | +9.71% | skill 1/1; base 1/1 |
| claude-code | compileiq-booster-pack-manifest-compiler-stages | 151,772 | 108,506 | +43,266 | +39.87% | skill 1/1; base 1/1 |
| claude-code | compileiq-booster-pack-negative-merge-intervals | 35,434 | 35,355 | +79 | +0.22% | skill 1/1; base 1/1 |
| claude-code | compileiq-booster-pack-negative-ptxas-register-spills | 36,581 | 36,380 | +201 | +0.55% | skill 1/1; base 1/1 |
| claude-code | compileiq-booster-pack-not-search-space-input | 224,554 | 183,462 | +41,092 | +22.40% | skill 1/1; base 1/1 |
| codex | All cases | 444,685 | 472,938 | -28,253 | -5.97% | skill 8/8; base 8/8 |
| codex | compileiq-booster-pack-apply-one-acf-mock-bench | 157,030 | 100,705 | +56,325 | +55.93% | skill 1/1; base 1/1 |
| codex | compileiq-booster-pack-canary-o0-no-regress | 38,614 | 24,733 | +13,881 | +56.12% | skill 1/1; base 1/1 |
| codex | compileiq-booster-pack-download-and-prereqs | 42,804 | 86,469 | -43,665 | -50.50% | skill 1/1; base 1/1 |
| codex | compileiq-booster-pack-flashinfer-batch-decode | 58,398 | 138,088 | -79,690 | -57.71% | skill 1/1; base 1/1 |
| codex | compileiq-booster-pack-manifest-compiler-stages | 78,066 | 42,151 | +35,915 | +85.21% | skill 1/1; base 1/1 |
| codex | compileiq-booster-pack-negative-merge-intervals | 13,532 | 13,360 | +172 | +1.29% | skill 1/1; base 1/1 |
| codex | compileiq-booster-pack-negative-ptxas-register-spills | 22,721 | 22,374 | +347 | +1.55% | skill 1/1; base 1/1 |
| codex | compileiq-booster-pack-not-search-space-input | 33,520 | 45,058 | -11,538 | -25.61% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,522,666 | 1,292,400 | +230,266 | +17.82% | skill 16/16; base 16/16 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 10 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/compileiq-booster-pack/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/compileiq-booster-pack/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/compileiq-booster-pack/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/compileiq-booster-pack/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (559 chars, recommend 50-150) (`skills/compileiq-booster-pack/SKILL.md`)
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
