# Skill Benchmark: parabricks

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `parabricks`
- Validation status: `passed`
- Evaluation date: 2026-10-27
- Evaluator version: `1.6.2`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 25 evaluation tasks (24 positive, 1 negative)
- Dataset digest: `sha256:f84268acba708284ca16102928430f9bc95d171ea7c78cb059a4922131f61f97` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| parabricks | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| parabricks | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| parabricks | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| parabricks | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |
| parabricks | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.1` | nondeterministic / provider_defaults |

- Controls for `parabricks`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 96.3% — baseline ran, but no comparable score was available; uplift unavailable | 87.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 96.0% → 98.0% (+2.0 points) | 96.0% → 96.0% (±0.0 points) |
| Correctness | 75.2% → 100.0% (+24.8 points) | 95.2% → 99.2% (+4.0 points) |
| Discoverability | 99.6% — baseline ran, but no comparable score was available; uplift unavailable | 92.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 57.2% → 94.8% (+37.6 points) | 76.8% → 84.4% (+7.6 points) |
| Efficiency | 89.0% — baseline ran, but no comparable score was available; uplift unavailable | 63.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 96.7% → 92.1% (-4.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 82.5% → 83.4% (+0.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,624,056 | 2,503,995 | +2,120,061 | +84.67% | skill 25/25; base 25/25 |
| claude-code | parabricks-001 | 121,270 | 110,773 | +10,497 | +9.48% | skill 1/1; base 1/1 |
| claude-code | parabricks-002 | 125,191 | 37,327 | +87,864 | +235.39% | skill 1/1; base 1/1 |
| claude-code | parabricks-003 | 123,259 | 38,377 | +84,882 | +221.18% | skill 1/1; base 1/1 |
| claude-code | parabricks-004 | 71,483 | 71,514 | -31 | -0.04% | skill 1/1; base 1/1 |
| claude-code | parabricks-005 | 186,693 | 37,001 | +149,692 | +404.56% | skill 1/1; base 1/1 |
| claude-code | parabricks-006 | 125,048 | 37,325 | +87,723 | +235.02% | skill 1/1; base 1/1 |
| claude-code | parabricks-007 | 119,710 | 37,110 | +82,600 | +222.58% | skill 1/1; base 1/1 |
| claude-code | parabricks-008 | 128,023 | 37,131 | +90,892 | +244.79% | skill 1/1; base 1/1 |
| claude-code | parabricks-009 | 122,937 | 38,601 | +84,336 | +218.48% | skill 1/1; base 1/1 |
| claude-code | parabricks-010 | 160,860 | 110,558 | +50,302 | +45.50% | skill 1/1; base 1/1 |
| claude-code | parabricks-011 | 165,354 | 37,663 | +127,691 | +339.04% | skill 1/1; base 1/1 |
| claude-code | parabricks-012 | 128,227 | 37,800 | +90,427 | +239.22% | skill 1/1; base 1/1 |
| claude-code | parabricks-013 | 254,319 | 39,138 | +215,181 | +549.80% | skill 1/1; base 1/1 |
| claude-code | parabricks-014 | 120,998 | 39,907 | +81,091 | +203.20% | skill 1/1; base 1/1 |
| claude-code | parabricks-015 | 254,768 | 40,161 | +214,607 | +534.37% | skill 1/1; base 1/1 |
| claude-code | parabricks-016 | 541,153 | 553,644 | -12,491 | -2.26% | skill 1/1; base 1/1 |
| claude-code | parabricks-017 | 416,184 | 425,284 | -9,100 | -2.14% | skill 1/1; base 1/1 |
| claude-code | parabricks-018 | 333,914 | 150,942 | +182,972 | +121.22% | skill 1/1; base 1/1 |
| claude-code | parabricks-019 | 168,077 | 145,334 | +22,743 | +15.65% | skill 1/1; base 1/1 |
| claude-code | parabricks-020 | 215,910 | 40,256 | +175,654 | +436.34% | skill 1/1; base 1/1 |
| claude-code | parabricks-021 | 172,401 | 146,482 | +25,919 | +17.69% | skill 1/1; base 1/1 |
| claude-code | parabricks-022 | 120,979 | 109,380 | +11,599 | +10.60% | skill 1/1; base 1/1 |
| claude-code | parabricks-023 | 161,731 | 109,575 | +52,156 | +47.60% | skill 1/1; base 1/1 |
| claude-code | parabricks-024 | 119,906 | 36,392 | +83,514 | +229.48% | skill 1/1; base 1/1 |
| claude-code | parabricks-025 | 165,661 | 36,320 | +129,341 | +356.12% | skill 1/1; base 1/1 |
| codex | All cases | 2,509,259 | 1,684,461 | +824,798 | +48.97% | skill 25/25; base 25/25 |
| codex | parabricks-001 | 131,283 | 347,370 | -216,087 | -62.21% | skill 1/1; base 1/1 |
| codex | parabricks-002 | 88,151 | 43,092 | +45,059 | +104.56% | skill 1/1; base 1/1 |
| codex | parabricks-003 | 171,611 | 93,035 | +78,576 | +84.46% | skill 1/1; base 1/1 |
| codex | parabricks-004 | 26,284 | 17,851 | +8,433 | +47.24% | skill 1/1; base 1/1 |
| codex | parabricks-005 | 118,936 | 38,789 | +80,147 | +206.62% | skill 1/1; base 1/1 |
| codex | parabricks-006 | 71,297 | 48,282 | +23,015 | +47.67% | skill 1/1; base 1/1 |
| codex | parabricks-007 | 53,368 | 26,892 | +26,476 | +98.45% | skill 1/1; base 1/1 |
| codex | parabricks-008 | 57,841 | 33,618 | +24,223 | +72.05% | skill 1/1; base 1/1 |
| codex | parabricks-009 | 108,528 | 51,435 | +57,093 | +111.00% | skill 1/1; base 1/1 |
| codex | parabricks-010 | 57,412 | 30,770 | +26,642 | +86.58% | skill 1/1; base 1/1 |
| codex | parabricks-011 | 53,401 | 26,263 | +27,138 | +103.33% | skill 1/1; base 1/1 |
| codex | parabricks-012 | 70,346 | 37,244 | +33,102 | +88.88% | skill 1/1; base 1/1 |
| codex | parabricks-013 | 146,547 | 47,793 | +98,754 | +206.63% | skill 1/1; base 1/1 |
| codex | parabricks-014 | 67,113 | 38,491 | +28,622 | +74.36% | skill 1/1; base 1/1 |
| codex | parabricks-015 | 56,359 | 73,165 | -16,806 | -22.97% | skill 1/1; base 1/1 |
| codex | parabricks-016 | 370,104 | 199,674 | +170,430 | +85.35% | skill 1/1; base 1/1 |
| codex | parabricks-017 | 287,837 | 195,415 | +92,422 | +47.30% | skill 1/1; base 1/1 |
| codex | parabricks-018 | 103,720 | 48,877 | +54,843 | +112.21% | skill 1/1; base 1/1 |
| codex | parabricks-019 | 76,755 | 46,821 | +29,934 | +63.93% | skill 1/1; base 1/1 |
| codex | parabricks-020 | 67,203 | 39,345 | +27,858 | +70.80% | skill 1/1; base 1/1 |
| codex | parabricks-021 | 96,989 | 46,269 | +50,720 | +109.62% | skill 1/1; base 1/1 |
| codex | parabricks-022 | 60,524 | 29,351 | +31,173 | +106.21% | skill 1/1; base 1/1 |
| codex | parabricks-023 | 53,297 | 30,129 | +23,168 | +76.90% | skill 1/1; base 1/1 |
| codex | parabricks-024 | 57,812 | 72,671 | -14,859 | -20.45% | skill 1/1; base 1/1 |
| codex | parabricks-025 | 56,541 | 21,819 | +34,722 | +159.14% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 7,133,315 | 4,188,456 | +2,944,859 | +70.31% | skill 50/50; base 50/50 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 25 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **CRITICAL** CONTENT_DEDUP/llm_error: LLM analysis failed for a content cluster (`skills/bionemo-agent-toolkit/skills/parabricks`)
- **MEDIUM** PII/ip_addresses: Public IP address (`references/shared-options.md:79`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/bionemo-agent-toolkit/skills/parabricks/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in pbrun-applybqsr.md (`skills/bionemo-agent-toolkit/skills/parabricks/SKILL.md`)
- **MEDIUM** SCHEMA/folder_hierarchy: Unexpected nesting depth for general skill (`skills/bionemo-agent-toolkit/skills/parabricks`)
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
