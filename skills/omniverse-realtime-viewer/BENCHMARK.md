# Skill Benchmark: omniverse-realtime-viewer

> ⚠️ **Overall verdict: INCOMPLETE — Required evidence is missing**

One or more required evaluation tiers did not complete, so this benchmark is not publication-complete.

## Evaluation Metadata

- Skill: `omniverse-realtime-viewer`
- Validation status: `incomplete`
- Evaluation date: 2026-11-06
- Evaluator version: `1.6.8`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 8 evaluation tasks (7 positive, 1 negative)
- Dataset digest: `sha256:b6a6c8ae48e6631377089dd22912257405540a3a1fecd6db54bdcce1e550ee79` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| omniverse-realtime-viewer | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| omniverse-realtime-viewer | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| omniverse-realtime-viewer | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| omniverse-realtime-viewer | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| omniverse-realtime-viewer | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `omniverse-realtime-viewer`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 92.7% — baseline ran, but no comparable score was available; uplift unavailable | 88.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 93.8% (-6.2 points) |
| Correctness | 45.0% → 97.5% (+52.5 points) | 82.5% → 100.0% (+17.5 points) |
| Discoverability | 97.9% — baseline ran, but no comparable score was available; uplift unavailable | 89.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 38.5% → 81.3% (+42.8 points) | 69.8% → 85.9% (+16.1 points) |
| Efficiency | 86.8% — baseline ran, but no comparable score was available; uplift unavailable | 75.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 88.6% → 84.6% (-4.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 84.7% → 82.3% (-2.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,653,260 | 740,861 | +1,912,399 | +258.13% | skill 8/8; base 8/8 |
| claude-code | unrelated-python-median | 35,207 | 35,149 | +58 | +0.17% | skill 1/1; base 1/1 |
| claude-code | viewer-browser-plan | 472,683 | 77,027 | +395,656 | +513.66% | skill 1/1; base 1/1 |
| claude-code | viewer-native-plan | 570,779 | 112,330 | +458,449 | +408.13% | skill 1/1; base 1/1 |
| claude-code | viewer-no-gpu | 348,886 | 108,488 | +240,398 | +221.59% | skill 1/1; base 1/1 |
| claude-code | viewer-nvcf-cleanup-plan | 397,138 | 110,455 | +286,683 | +259.55% | skill 1/1; base 1/1 |
| claude-code | viewer-pick-and-normal-review | 268,917 | 110,226 | +158,691 | +143.97% | skill 1/1; base 1/1 |
| claude-code | viewer-scene-replacement | 353,685 | 112,013 | +241,672 | +215.75% | skill 1/1; base 1/1 |
| claude-code | viewer-source-metadata | 205,965 | 75,173 | +130,792 | +173.99% | skill 1/1; base 1/1 |
| codex | All cases | 1,693,725 | 524,841 | +1,168,884 | +222.71% | skill 8/8; base 8/8 |
| codex | unrelated-python-median | 13,316 | 13,260 | +56 | +0.42% | skill 1/1; base 1/1 |
| codex | viewer-browser-plan | 99,011 | 40,599 | +58,412 | +143.88% | skill 1/1; base 1/1 |
| codex | viewer-native-plan | 125,971 | 50,473 | +75,498 | +149.58% | skill 1/1; base 1/1 |
| codex | viewer-no-gpu | 74,574 | 45,261 | +29,313 | +64.76% | skill 1/1; base 1/1 |
| codex | viewer-nvcf-cleanup-plan | 884,773 | 209,438 | +675,335 | +322.45% | skill 1/1; base 1/1 |
| codex | viewer-pick-and-normal-review | 181,148 | 83,514 | +97,634 | +116.91% | skill 1/1; base 1/1 |
| codex | viewer-scene-replacement | 168,625 | 41,657 | +126,968 | +304.79% | skill 1/1; base 1/1 |
| codex | viewer-source-metadata | 146,307 | 40,639 | +105,668 | +260.02% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 4,346,985 | 1,265,702 | +3,081,283 | +243.44% | skill 16/16; base 16/16 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 71 finding(s) |
| Tier 2 | Semantic deduplication | **INCOMPLETE** | Tier 2 scan incomplete: produced more than 1024 content chunks. |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Incomplete Coverage

- **Context Deduplication:** Tier 2 scan incomplete: produced more than 1024 content chunks.

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** PII/crypto_addresses: Bitcoin address (`references/cloud-deployment/README.md:21`)
- **MEDIUM** PII/phone_numbers: International phone number (`references/stylized-visualization/style-recipes.md:133`)
- **MEDIUM** PII/phone_numbers: International phone number (`references/stylized-visualization/style-recipes.md:137`)
- **MEDIUM** PII/phone_numbers: International phone number (`references/stylized-visualization/style-recipes.md:138`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in conventions.md (`skills/omniverse-realtime-viewer/SKILL.md`)
- 66 additional finding(s) are available in the full evaluation artifacts.

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
