# Skill Benchmark: compileiq-search-space

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `compileiq-search-space`
- Validation status: `passed`
- Evaluation date: 2026-11-14
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 8 evaluation tasks (6 positive, 2 negative)
- Dataset digest: `sha256:6a725db989ce9bc944391c21a818c2dfae298e1dc37867df583f1d2c11c60db7` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| compileiq-search-space | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-search-space | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-search-space | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-search-space | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| compileiq-search-space | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `compileiq-search-space`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 95.5% — baseline ran, but no comparable score was available; uplift unavailable | 87.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 93.8% → 100.0% (+6.2 points) |
| Correctness | 62.5% → 100.0% (+37.5 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 42.2% → 96.9% (+54.7 points) | 75.4% → 85.3% (+9.9 points) |
| Efficiency | 80.4% — baseline ran, but no comparable score was available; uplift unavailable | 59.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.2% → 97.0% (-0.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 85.0% → 87.6% (+2.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 830,164 | 1,177,670 | -347,506 | -29.51% | skill 8/8; base 8/8 |
| claude-code | compileiq-search-space-airgapped | 111,906 | 381,675 | -269,769 | -70.68% | skill 1/1; base 1/1 |
| claude-code | compileiq-search-space-att-variant | 113,663 | 145,616 | -31,953 | -21.94% | skill 1/1; base 1/1 |
| claude-code | compileiq-search-space-custom-mixed | 154,767 | 144,808 | +9,959 | +6.88% | skill 1/1; base 1/1 |
| claude-code | compileiq-search-space-negative-largest-files | 35,133 | 35,240 | -107 | -0.30% | skill 1/1; base 1/1 |
| claude-code | compileiq-search-space-negative-maxrregcount | 35,968 | 36,079 | -111 | -0.31% | skill 1/1; base 1/1 |
| claude-code | compileiq-search-space-pick-provider | 112,232 | 144,386 | -32,154 | -22.27% | skill 1/1; base 1/1 |
| claude-code | compileiq-search-space-pin-release | 153,671 | 144,538 | +9,133 | +6.32% | skill 1/1; base 1/1 |
| claude-code | compileiq-search-space-review-script | 112,824 | 145,328 | -32,504 | -22.37% | skill 1/1; base 1/1 |
| codex | All cases | 324,249 | 509,455 | -185,206 | -36.35% | skill 8/8; base 8/8 |
| codex | compileiq-search-space-airgapped | 40,330 | 41,971 | -1,641 | -3.91% | skill 1/1; base 1/1 |
| codex | compileiq-search-space-att-variant | 37,908 | 24,404 | +13,504 | +55.34% | skill 1/1; base 1/1 |
| codex | compileiq-search-space-custom-mixed | 58,896 | 104,682 | -45,786 | -43.74% | skill 1/1; base 1/1 |
| codex | compileiq-search-space-negative-largest-files | 13,515 | 13,382 | +133 | +0.99% | skill 1/1; base 1/1 |
| codex | compileiq-search-space-negative-maxrregcount | 36,544 | 22,172 | +14,372 | +64.82% | skill 1/1; base 1/1 |
| codex | compileiq-search-space-pick-provider | 38,355 | 41,904 | -3,549 | -8.47% | skill 1/1; base 1/1 |
| codex | compileiq-search-space-pin-release | 40,075 | 172,531 | -132,456 | -76.77% | skill 1/1; base 1/1 |
| codex | compileiq-search-space-review-script | 58,626 | 88,409 | -29,783 | -33.69% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,154,413 | 1,687,125 | -532,712 | -31.58% | skill 16/16; base 16/16 |

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

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/compileiq-search-space/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/compileiq-search-space/SKILL.md`)
- **MEDIUM** SECURITY/Session Persistence (RA2): Rogue Agent: mkdir -p /shared/ciq-search-spaces
gh release download "$SS_TAG" -R NVIDIA/CompileIQ -D /shared/ciq-search-spaces

# Move the directory to the air-gapped host (rsync, scp, sneaker-net, ...)

# On the  (`SKILL.md:103`)
- **MEDIUM** SECURITY/Session Persistence (RA2): Rogue Agent: mkdir -p /shared/ciq-search-spaces
gh release download "$SS_TAG" -R NVIDIA/CompileIQ -D /shared/ciq-search-spaces

# Move the directory to the air-gapped host (rsync, scp, sneaker-net, …)

# On the ai (`SKILL.md:103`)
- **LOW** QUALITY/quality_discoverability: Description very long (608 chars, recommend 50-150) (`skills/compileiq-search-space/SKILL.md`)
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
