# Skill Benchmark: nvidia-skill-finder

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvidia-skill-finder`
- Validation status: `passed`
- Evaluation date: 2026-11-04
- Evaluator version: `1.6.6`
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
| nvidia-skill-finder | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| nvidia-skill-finder | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| nvidia-skill-finder | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| nvidia-skill-finder | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| nvidia-skill-finder | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.3` | nondeterministic / provider_defaults |

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
| Overall | 94.1% — baseline ran, but no comparable score was available; uplift unavailable | 87.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 97.2% → 97.2% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 62.2% → 96.7% (+34.5 points) | 77.8% → 91.1% (+13.3 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 90.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 39.9% → 96.5% (+56.6 points) | 59.3% → 92.1% (+32.8 points) |
| Efficiency | 80.3% — baseline ran, but no comparable score was available; uplift unavailable | 61.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.3% → 96.4% (-1.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 90.0% → 88.9% (-1.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,746,750 | 1,692,842 | +1,053,908 | +62.26% | skill 18/18; base 18/18 |
| claude-code | nvidia-skill-finder-diff-bare-invocation | 158,485 | 70,559 | +87,926 | +124.61% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-diff-browse-no-keywords | 160,553 | 107,298 | +53,255 | +49.63% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-diff-deep-research | 153,825 | 35,323 | +118,502 | +335.48% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-express-route | 224,006 | 182,756 | +41,250 | +22.57% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-generic-kubernetes | 109,046 | 142,661 | -33,615 | -23.56% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-python-refactor | 107,662 | 106,575 | +1,087 | +1.02% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-react-optimize | 144,923 | 107,108 | +37,815 | +35.31% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-video-editing | 108,246 | 71,223 | +37,023 | +51.98% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-cad-to-simready | 153,252 | 143,560 | +9,692 | +6.75% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-dicom | 154,197 | 35,935 | +118,262 | +329.10% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-dynamo-kv | 195,216 | 144,272 | +50,944 | +35.31% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-gpu-pandas | 153,152 | 107,398 | +45,754 | +42.60% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-isaac-lab | 154,942 | 36,723 | +118,219 | +321.92% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-jetson-driver-install | 153,508 | 35,926 | +117,582 | +327.29% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-jetson-setup | 153,903 | 107,200 | +46,703 | +43.57% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-multigpu-training | 154,782 | 151,902 | +2,880 | +1.90% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-openusd-optimization | 153,925 | 70,956 | +82,969 | +116.93% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-vehicle-routing | 153,127 | 35,467 | +117,660 | +331.75% | skill 1/1; base 1/1 |
| codex | All cases | 1,130,804 | 610,948 | +519,856 | +85.09% | skill 18/18; base 18/18 |
| codex | nvidia-skill-finder-diff-bare-invocation | 68,890 | 35,919 | +32,971 | +91.79% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-diff-browse-no-keywords | 89,599 | 52,784 | +36,815 | +69.75% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-diff-deep-research | 90,172 | 35,866 | +54,306 | +151.41% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-express-route | 55,483 | 40,948 | +14,535 | +35.50% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-generic-kubernetes | 41,915 | 27,157 | +14,758 | +54.34% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-python-refactor | 41,302 | 40,437 | +865 | +2.14% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-react-optimize | 40,956 | 41,097 | -141 | -0.34% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-video-editing | 13,505 | 13,316 | +189 | +1.42% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-cad-to-simready | 67,881 | 13,365 | +54,516 | +407.90% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-dicom | 73,056 | 38,531 | +34,525 | +89.60% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-dynamo-kv | 17,983 | 31,520 | -13,537 | -42.95% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-gpu-pandas | 69,058 | 22,021 | +47,037 | +213.60% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-isaac-lab | 106,097 | 56,158 | +49,939 | +88.93% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-jetson-driver-install | 69,157 | 17,804 | +51,353 | +288.44% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-jetson-setup | 69,085 | 17,655 | +51,430 | +291.31% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-multigpu-training | 77,888 | 22,466 | +55,422 | +246.69% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-openusd-optimization | 68,381 | 65,091 | +3,290 | +5.05% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-vehicle-routing | 70,396 | 38,813 | +31,583 | +81.37% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 3,877,554 | 2,303,790 | +1,573,764 | +68.31% | skill 36/36; base 36/36 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 14 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 18 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`plugins/nvidia-skills/skills/nvidia-skill-finder/SKILL.md`)
- **MEDIUM** SECURITY/MCP Rug Pull (RP1): MCP Rug Pull: npx commands without a version suffix (e.g. @1.0.0) create a rug-pull risk if the upstream server is compromised and pub (`SKILL.md:74`)
- **MEDIUM** SECURITY/MCP Rug Pull (RP1): MCP Rug Pull: npx commands without a version suffix (e.g. @1.0.0) create a rug-pull risk if the upstream server is compromised and pub (`SKILL.md:135`)
- **MEDIUM** SECURITY/MCP Rug Pull (RP1): MCP Rug Pull: npx commands without a version suffix (e.g. @1.0.0) create a rug-pull risk if the upstream server is compromised and pub (`SKILL.md:103`)
- **MEDIUM** SECURITY/MCP Rug Pull (RP1): MCP Rug Pull: npx commands without a version suffix (e.g. @1.0.0) create a rug-pull risk if the upstream server is compromised and pub (`SKILL.md:148`)
- 9 additional finding(s) are available in the full evaluation artifacts.

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
