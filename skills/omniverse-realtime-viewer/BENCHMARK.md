# Skill Benchmark: omniverse-realtime-viewer

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `omniverse-realtime-viewer`
- Validation status: `passed`
- Evaluation date: 2026-11-06
- Evaluator version: `1.6.9`
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
| Overall | 93.9% — baseline ran, but no comparable score was available; uplift unavailable | 88.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 93.8% (-6.2 points) |
| Correctness | 42.5% → 100.0% (+57.5 points) | 80.0% → 100.0% (+20.0 points) |
| Discoverability | 93.6% — baseline ran, but no comparable score was available; uplift unavailable | 91.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 54.2% → 89.1% (+34.9 points) | 66.7% → 82.3% (+15.6 points) |
| Efficiency | 86.9% — baseline ran, but no comparable score was available; uplift unavailable | 72.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 92.3% → 86.0% (-6.3 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 84.7% → 81.8% (-2.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,843,994 | 735,099 | +2,108,895 | +286.89% | skill 8/8; base 8/8 |
| claude-code | unrelated-python-median | 35,302 | 35,140 | +162 | +0.46% | skill 1/1; base 1/1 |
| claude-code | viewer-browser-plan | 496,313 | 75,284 | +421,029 | +559.25% | skill 1/1; base 1/1 |
| claude-code | viewer-native-plan | 758,174 | 74,747 | +683,427 | +914.32% | skill 1/1; base 1/1 |
| claude-code | viewer-no-gpu | 282,641 | 109,197 | +173,444 | +158.84% | skill 1/1; base 1/1 |
| claude-code | viewer-nvcf-cleanup-plan | 358,536 | 109,072 | +249,464 | +228.71% | skill 1/1; base 1/1 |
| claude-code | viewer-pick-and-normal-review | 332,760 | 109,120 | +223,640 | +204.95% | skill 1/1; base 1/1 |
| claude-code | viewer-scene-replacement | 361,339 | 112,163 | +249,176 | +222.16% | skill 1/1; base 1/1 |
| claude-code | viewer-source-metadata | 218,929 | 110,376 | +108,553 | +98.35% | skill 1/1; base 1/1 |
| codex | All cases | 1,388,192 | 488,082 | +900,110 | +184.42% | skill 8/8; base 8/8 |
| codex | unrelated-python-median | 13,317 | 13,253 | +64 | +0.48% | skill 1/1; base 1/1 |
| codex | viewer-browser-plan | 143,609 | 49,170 | +94,439 | +192.07% | skill 1/1; base 1/1 |
| codex | viewer-native-plan | 99,695 | 58,871 | +40,824 | +69.34% | skill 1/1; base 1/1 |
| codex | viewer-no-gpu | 75,062 | 58,192 | +16,870 | +28.99% | skill 1/1; base 1/1 |
| codex | viewer-nvcf-cleanup-plan | 656,841 | 110,393 | +546,448 | +495.00% | skill 1/1; base 1/1 |
| codex | viewer-pick-and-normal-review | 188,428 | 98,380 | +90,048 | +91.53% | skill 1/1; base 1/1 |
| codex | viewer-scene-replacement | 125,034 | 41,679 | +83,355 | +199.99% | skill 1/1; base 1/1 |
| codex | viewer-source-metadata | 86,206 | 58,144 | +28,062 | +48.26% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 4,232,186 | 1,223,181 | +3,009,005 | +246.00% | skill 16/16; base 16/16 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 70 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 54 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/aov-switching/README.md and references/ovrtx-rendering/README.md and references/usd-viewer-app/README.md:
  "## Processed GPU outputs" in references/aov-switching/README.md (lines 206-213)
  vs "## Processed GPU outputs" in references/ovrtx-rendering/README.md (lines 433-440)
  vs "## Processed GPU outputs" in references/usd-viewer-app/README.md (lines 91-98) (`references/aov-switching/README.md:206`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across SKILL.md and references/dependencies/README.md and references/routing.md:
  "## Solid Modeling: Upstream First" in SKILL.md (lines 67-76)
  vs "## USD BRep: Public Solid Modeling Dependency" in references/dependencies/README.md (lines 77-108)
  vs "## Solid Modeling: Upstream First" in references/routing.md (lines 349-362) (`SKILL.md:67`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/cpp-native-viewer/interaction-features.md and references/native-picking-selection/README.md and references/object-selection/README.md:
  "## Native Picking" in references/cpp-native-viewer/interaction-features.md (lines 92-93)
  vs "## Native Picking" in references/cpp-native-viewer/interaction-features.md (lines 173-191)
  vs "## Decode Pick Results" in references/native-picking-selection/README.md (lines 144-183)
  vs "## Decode Pick Hits" in references/object-selection/README.md (lines 150-187) (`references/cpp-native-viewer/interaction-features.md:92`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/dependencies/environment-validation.md and references/dependencies/ovrtx.md and references/ovrtx-rendering/README.md and references/ovui-local-viewer-recipe/setup-shell-renderer.md and references/streaming-viewer-recipe/server-runtime.md and references/troubleshooting/README.md:
  "## Environment Variable Summary" in references/dependencies/environment-validation.md (lines 3-43)
  vs "## ovrtx" in references/dependencies/ovrtx.md (lines 32-37)
  vs "## ovrtx" in references/dependencies/ovrtx.md (lines 43-51)
  vs "## Environment" in references/ovrtx-rendering/README.md (lines 338-347)
  vs "## Common Errors" in references/ovrtx-rendering/README.md (lines 408-432)
  vs "## 2. Install Dependencies And Configure The Environment" in references/ovui-local-viewer-recipe/setup-shell-renderer.md (lines 29-33)
  vs "## 2. Install Dependencies And Configure The Environment" in references/streaming-viewer-recipe/server-runtime.md (lines 43-49)
  vs "## Common Error Map" in references/troubleshooting/README.md (lines 63-101) (`references/dependencies/environment-validation.md:3`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/ovui-local-viewer-recipe/interaction-features.md and references/render-settings/README.md and references/streaming-viewer-recipe/interaction-features.md:
  "## 11. Add Render Settings And Lighting Controls" in references/ovui-local-viewer-recipe/interaction-features.md (lines 190-196)
  vs "## Capability-Driven Settings" in references/render-settings/README.md (lines 22-54)
  vs "## 13. Add Render Settings And Lighting Controls" in references/streaming-viewer-recipe/interaction-features.md (lines 172-178) (`references/ovui-local-viewer-recipe/interaction-features.md:190`)
- 119 additional finding(s) are available in the full evaluation artifacts.

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
