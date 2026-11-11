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
| Overall | 94.2% — baseline ran, but no comparable score was available; uplift unavailable | 89.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 97.2% → 97.2% (±0.0 points) | 100.0% → 91.7% (-8.3 points) |
| Correctness | 64.4% → 96.7% (+32.3 points) | 74.4% → 96.7% (+22.3 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 36.4% → 95.8% (+59.4 points) | 58.6% → 98.6% (+40.0 points) |
| Efficiency | 81.3% — baseline ran, but no comparable score was available; uplift unavailable | 64.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.2% → 96.4% (-1.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 90.3% → 88.9% (-1.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,745,616 | 1,880,457 | +865,159 | +46.01% | skill 18/18; base 18/18 |
| claude-code | nvidia-skill-finder-diff-bare-invocation | 158,578 | 107,454 | +51,124 | +47.58% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-diff-browse-no-keywords | 160,580 | 71,104 | +89,476 | +125.84% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-diff-deep-research | 153,995 | 35,277 | +118,718 | +336.53% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-express-route | 149,387 | 221,515 | -72,128 | -32.56% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-generic-kubernetes | 108,200 | 143,204 | -35,004 | -24.44% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-python-refactor | 143,192 | 141,724 | +1,468 | +1.04% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-react-optimize | 144,879 | 106,899 | +37,980 | +35.53% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-video-editing | 107,858 | 107,020 | +838 | +0.78% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-cad-to-simready | 153,258 | 143,526 | +9,732 | +6.78% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-dicom | 153,494 | 35,674 | +117,820 | +330.27% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-dynamo-kv | 233,649 | 144,374 | +89,275 | +61.84% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-gpu-pandas | 154,289 | 107,412 | +46,877 | +43.64% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-isaac-lab | 154,854 | 72,465 | +82,389 | +113.69% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-jetson-driver-install | 153,954 | 35,683 | +118,271 | +331.45% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-jetson-setup | 153,470 | 107,265 | +46,205 | +43.08% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-multigpu-training | 154,720 | 193,877 | -39,157 | -20.20% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-openusd-optimization | 154,034 | 70,646 | +83,388 | +118.04% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-vehicle-routing | 153,225 | 35,338 | +117,887 | +333.60% | skill 1/1; base 1/1 |
| codex | All cases | 1,217,904 | 634,064 | +583,840 | +92.08% | skill 18/18; base 18/18 |
| codex | nvidia-skill-finder-diff-bare-invocation | 81,560 | 35,508 | +46,052 | +129.69% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-diff-browse-no-keywords | 85,718 | 74,582 | +11,136 | +14.93% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-diff-deep-research | 86,248 | 41,228 | +45,020 | +109.20% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-express-route | 41,182 | 54,623 | -13,441 | -24.61% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-generic-kubernetes | 70,198 | 41,207 | +28,991 | +70.35% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-python-refactor | 40,807 | 40,749 | +58 | +0.14% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-react-optimize | 41,069 | 40,473 | +596 | +1.47% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-video-editing | 13,401 | 13,277 | +124 | +0.93% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-cad-to-simready | 67,896 | 13,288 | +54,608 | +410.96% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-dicom | 110,655 | 23,439 | +87,216 | +372.10% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-dynamo-kv | 55,280 | 17,709 | +37,571 | +212.16% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-gpu-pandas | 69,086 | 22,158 | +46,928 | +211.79% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-isaac-lab | 99,343 | 49,006 | +50,337 | +102.72% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-jetson-driver-install | 73,473 | 17,650 | +55,823 | +316.28% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-jetson-setup | 68,993 | 17,646 | +51,347 | +290.98% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-multigpu-training | 73,752 | 22,455 | +51,297 | +228.44% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-openusd-optimization | 68,934 | 65,372 | +3,562 | +5.45% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-vehicle-routing | 70,309 | 43,694 | +26,615 | +60.91% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 3,963,520 | 2,514,521 | +1,448,999 | +57.63% | skill 36/36; base 36/36 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 15 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 18 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nvidia-skill-finder/SKILL.md`)
- **MEDIUM** SECURITY/MCP Rug Pull (RP1): MCP Rug Pull: npx commands without a version suffix (e.g. @1.0.0) create a rug-pull risk if the upstream server is compromised and pub (`SKILL.md:74`)
- **MEDIUM** SECURITY/MCP Rug Pull (RP1): MCP Rug Pull: npx commands without a version suffix (e.g. @1.0.0) create a rug-pull risk if the upstream server is compromised and pub (`SKILL.md:103`)
- **MEDIUM** SECURITY/MCP Rug Pull (RP1): MCP Rug Pull: npx commands without a version suffix (e.g. @1.0.0) create a rug-pull risk if the upstream server is compromised and pub (`SKILL.md:183`)
- **MEDIUM** SECURITY/MCP Rug Pull (RP1): MCP Rug Pull: npx commands without a version suffix (e.g. @1.0.0) create a rug-pull risk if the upstream server is compromised and pub (`SKILL.md:113`)
- 10 additional finding(s) are available in the full evaluation artifacts.

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
