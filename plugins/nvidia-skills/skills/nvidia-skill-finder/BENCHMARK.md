# Skill Benchmark: nvidia-skill-finder

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nvidia-skill-finder`
- Validation status: `passed`
- Evaluation date: 2026-11-05
- Evaluator version: `1.6.8`
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
| Overall | 94.0% — baseline ran, but no comparable score was available; uplift unavailable | 89.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 97.2% → 100.0% (+2.8 points) | 100.0% → 94.4% (-5.6 points) |
| Correctness | 61.1% → 94.4% (+33.3 points) | 74.4% → 93.3% (+18.9 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 36.5% → 96.5% (+60.0 points) | 56.5% → 100.0% (+43.5 points) |
| Efficiency | 79.2% — baseline ran, but no comparable score was available; uplift unavailable | 63.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.4% → 93.5% (-4.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 90.3% → 88.3% (-2.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,563,597 | 1,758,243 | +805,354 | +45.80% | skill 18/18; base 18/18 |
| claude-code | nvidia-skill-finder-diff-bare-invocation | 163,746 | 143,042 | +20,704 | +14.47% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-diff-browse-no-keywords | 161,047 | 141,786 | +19,261 | +13.58% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-diff-deep-research | 153,874 | 35,301 | +118,573 | +335.89% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-express-route | 71,576 | 220,779 | -149,203 | -67.58% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-generic-kubernetes | 107,929 | 107,725 | +204 | +0.19% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-python-refactor | 70,858 | 106,486 | -35,628 | -33.46% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-react-optimize | 108,001 | 106,758 | +1,243 | +1.16% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-video-editing | 107,860 | 71,136 | +36,724 | +51.63% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-cad-to-simready | 153,429 | 107,060 | +46,369 | +43.31% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-dicom | 153,857 | 35,665 | +118,192 | +331.39% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-dynamo-kv | 233,877 | 144,043 | +89,834 | +62.37% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-gpu-pandas | 153,226 | 107,361 | +45,865 | +42.72% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-isaac-lab | 154,854 | 72,468 | +82,386 | +113.69% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-jetson-driver-install | 153,460 | 35,709 | +117,751 | +329.75% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-jetson-setup | 153,472 | 107,158 | +46,314 | +43.22% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-multigpu-training | 155,620 | 109,524 | +46,096 | +42.09% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-openusd-optimization | 153,832 | 70,893 | +82,939 | +116.99% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-vehicle-routing | 153,079 | 35,349 | +117,730 | +333.05% | skill 1/1; base 1/1 |
| codex | All cases | 1,203,358 | 676,098 | +527,260 | +77.99% | skill 18/18; base 18/18 |
| codex | nvidia-skill-finder-diff-bare-invocation | 68,816 | 39,893 | +28,923 | +72.50% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-diff-browse-no-keywords | 52,454 | 78,017 | -25,563 | -32.77% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-diff-deep-research | 86,657 | 36,197 | +50,460 | +139.40% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-express-route | 41,182 | 54,920 | -13,738 | -25.01% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-generic-kubernetes | 41,660 | 27,124 | +14,536 | +53.59% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-python-refactor | 41,277 | 40,337 | +940 | +2.33% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-react-optimize | 41,082 | 68,215 | -27,133 | -39.78% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-video-editing | 13,432 | 13,353 | +79 | +0.59% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-cad-to-simready | 67,885 | 13,304 | +54,581 | +410.26% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-dicom | 74,804 | 30,361 | +44,443 | +146.38% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-dynamo-kv | 71,043 | 41,039 | +30,004 | +73.11% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-gpu-pandas | 69,037 | 22,219 | +46,818 | +210.71% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-isaac-lab | 156,536 | 52,466 | +104,070 | +198.36% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-jetson-driver-install | 73,541 | 17,722 | +55,819 | +314.97% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-jetson-setup | 53,484 | 17,795 | +35,689 | +200.56% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-multigpu-training | 73,674 | 22,164 | +51,510 | +232.40% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-openusd-optimization | 68,956 | 65,081 | +3,875 | +5.95% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-vehicle-routing | 107,838 | 35,891 | +71,947 | +200.46% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 3,766,955 | 2,434,341 | +1,332,614 | +54.74% | skill 36/36; base 36/36 |

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
- **MEDIUM** SECURITY/MCP Rug Pull (RP1): MCP Rug Pull: npx commands without a version suffix (e.g. @1.0.0) create a rug-pull risk if the upstream server is compromised and pub (`SKILL.md:103`)
- **MEDIUM** SECURITY/MCP Rug Pull (RP1): MCP Rug Pull: npx commands without a version suffix (e.g. @1.0.0) create a rug-pull risk if the upstream server is compromised and pub (`SKILL.md:113`)
- **MEDIUM** SECURITY/MCP Rug Pull (RP1): MCP Rug Pull: npx commands without a version suffix (e.g. @1.0.0) create a rug-pull risk if the upstream server is compromised and pub (`SKILL.md:135`)
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
