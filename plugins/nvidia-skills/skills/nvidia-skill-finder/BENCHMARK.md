# Skill Benchmark: nvidia-skill-finder

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvidia-skill-finder`
- Validation status: `passed`
- Evaluation date: 2026-11-10
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 18 evaluation tasks (13 positive, 5 negative)
- Dataset digest: `sha256:75b2b97f4c0057ea5bbadcfcd92d5561919ce56e7f03a93f629cb8bdd8f01281` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nvidia-skill-finder | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvidia-skill-finder | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvidia-skill-finder | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvidia-skill-finder | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| nvidia-skill-finder | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `nvidia-skill-finder`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 95.0% — baseline ran, but no comparable score was available; uplift unavailable | 87.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 97.2% → 100.0% (+2.8 points) | 97.2% → 91.7% (-5.5 points) |
| Correctness | 63.3% → 96.7% (+33.4 points) | 77.8% → 93.3% (+15.5 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 94.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 37.1% → 96.5% (+59.4 points) | 59.5% → 94.4% (+34.9 points) |
| Efficiency | 81.8% — baseline ran, but no comparable score was available; uplift unavailable | 65.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.5% → 96.3% (-2.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 89.4% → 89.2% (-0.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,711,937 | 1,615,131 | +1,096,806 | +67.91% | skill 18/18; base 18/18 |
| claude-code | nvidia-skill-finder-diff-bare-invocation | 158,444 | 70,725 | +87,719 | +124.03% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-diff-browse-no-keywords | 159,838 | 70,753 | +89,085 | +125.91% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-diff-deep-research | 159,845 | 35,205 | +124,640 | +354.04% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-express-route | 225,059 | 183,972 | +41,087 | +22.33% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-generic-kubernetes | 108,443 | 107,258 | +1,185 | +1.10% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-python-refactor | 70,868 | 178,071 | -107,203 | -60.20% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-react-optimize | 145,172 | 143,073 | +2,099 | +1.47% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-video-editing | 108,172 | 71,109 | +37,063 | +52.12% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-cad-to-simready | 153,814 | 106,978 | +46,836 | +43.78% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-dicom | 153,832 | 70,874 | +82,958 | +117.05% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-dynamo-kv | 191,996 | 144,429 | +47,567 | +32.93% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-gpu-pandas | 153,323 | 108,233 | +45,090 | +41.66% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-isaac-lab | 154,640 | 37,221 | +117,419 | +315.46% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-jetson-driver-install | 153,527 | 35,927 | +117,600 | +327.33% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-jetson-setup | 153,277 | 71,735 | +81,542 | +113.67% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-multigpu-training | 154,876 | 108,772 | +46,104 | +42.39% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-openusd-optimization | 153,774 | 35,357 | +118,417 | +334.92% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-vehicle-routing | 153,037 | 35,439 | +117,598 | +331.83% | skill 1/1; base 1/1 |
| codex | All cases | 1,183,079 | 708,839 | +474,240 | +66.90% | skill 18/18; base 18/18 |
| codex | nvidia-skill-finder-diff-bare-invocation | 82,443 | 39,981 | +42,462 | +106.21% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-diff-browse-no-keywords | 74,467 | 86,422 | -11,955 | -13.83% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-diff-deep-research | 74,436 | 58,872 | +15,564 | +26.44% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-express-route | 40,960 | 54,898 | -13,938 | -25.39% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-generic-kubernetes | 41,583 | 27,181 | +14,402 | +52.99% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-python-refactor | 41,193 | 40,300 | +893 | +2.22% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-react-optimize | 55,211 | 40,942 | +14,269 | +34.85% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-video-editing | 13,438 | 13,250 | +188 | +1.42% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-cad-to-simready | 52,176 | 17,520 | +34,656 | +197.81% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-dicom | 71,768 | 30,224 | +41,544 | +137.45% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-dynamo-kv | 55,355 | 44,962 | +10,393 | +23.12% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-gpu-pandas | 68,989 | 17,840 | +51,149 | +286.71% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-isaac-lab | 99,264 | 69,570 | +29,694 | +42.68% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-jetson-driver-install | 69,382 | 17,671 | +51,711 | +292.63% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-jetson-setup | 69,017 | 17,581 | +51,436 | +292.57% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-multigpu-training | 57,797 | 30,750 | +27,047 | +87.96% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-openusd-optimization | 145,160 | 65,166 | +79,994 | +122.75% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-vehicle-routing | 70,440 | 35,709 | +34,731 | +97.26% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 3,895,016 | 2,323,970 | +1,571,046 | +67.60% | skill 36/36; base 36/36 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 16 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 18 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`plugins/nvidia-skills/skills/nvidia-skill-finder/SKILL.md`)
- **MEDIUM** SECURITY/MCP Rug Pull (RP1): MCP Rug Pull: npx commands without a version suffix (e.g. @1.0.0) create a rug-pull risk if the upstream server is compromised and pub (`SKILL.md:74`)
- **MEDIUM** SECURITY/MCP Rug Pull (RP1): MCP Rug Pull: npx commands without a version suffix (e.g. @1.0.0) create a rug-pull risk if the upstream server is compromised and pub (`SKILL.md:103`)
- **MEDIUM** SECURITY/MCP Rug Pull (RP1): MCP Rug Pull: npx commands without a version suffix (e.g. @1.0.0) create a rug-pull risk if the upstream server is compromised and pub (`SKILL.md:135`)
- **MEDIUM** SECURITY/MCP Rug Pull (RP1): MCP Rug Pull: npx commands without a version suffix (e.g. @1.0.0) create a rug-pull risk if the upstream server is compromised and pub (`SKILL.md:113`)
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
