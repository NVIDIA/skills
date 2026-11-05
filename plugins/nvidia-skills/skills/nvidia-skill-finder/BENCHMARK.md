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
| Overall | 93.4% — baseline ran, but no comparable score was available; uplift unavailable | 87.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 97.2% (-2.8 points) | 97.2% → 91.7% (-5.5 points) |
| Correctness | 64.4% → 96.7% (+32.3 points) | 77.8% → 91.1% (+13.3 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 37.1% → 93.8% (+56.7 points) | 58.8% → 94.2% (+35.4 points) |
| Efficiency | 79.6% — baseline ran, but no comparable score was available; uplift unavailable | 64.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.3% → 96.2% (-2.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 89.3% → 88.2% (-1.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,940,059 | 1,733,228 | +1,206,831 | +69.63% | skill 18/18; base 18/18 |
| claude-code | nvidia-skill-finder-diff-bare-invocation | 159,556 | 107,051 | +52,505 | +49.05% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-diff-browse-no-keywords | 160,640 | 106,788 | +53,852 | +50.43% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-diff-deep-research | 349,223 | 35,273 | +313,950 | +890.06% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-express-route | 185,632 | 223,103 | -37,471 | -16.80% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-generic-kubernetes | 72,345 | 71,645 | +700 | +0.98% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-python-refactor | 142,972 | 141,956 | +1,016 | +0.72% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-react-optimize | 145,032 | 142,851 | +2,181 | +1.53% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-video-editing | 107,845 | 71,165 | +36,680 | +51.54% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-cad-to-simready | 154,290 | 143,642 | +10,648 | +7.41% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-dicom | 154,191 | 35,577 | +118,614 | +333.40% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-dynamo-kv | 231,606 | 144,237 | +87,369 | +60.57% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-gpu-pandas | 153,338 | 108,293 | +45,045 | +41.60% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-isaac-lab | 154,754 | 72,321 | +82,433 | +113.98% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-jetson-driver-install | 153,792 | 35,680 | +118,112 | +331.03% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-jetson-setup | 153,387 | 71,488 | +81,899 | +114.56% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-multigpu-training | 154,938 | 151,301 | +3,637 | +2.40% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-openusd-optimization | 153,414 | 35,400 | +118,014 | +333.37% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-vehicle-routing | 153,104 | 35,457 | +117,647 | +331.80% | skill 1/1; base 1/1 |
| codex | All cases | 1,208,315 | 806,320 | +401,995 | +49.86% | skill 18/18; base 18/18 |
| codex | nvidia-skill-finder-diff-bare-invocation | 86,372 | 35,913 | +50,459 | +140.50% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-diff-browse-no-keywords | 93,587 | 47,277 | +46,310 | +97.95% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-diff-deep-research | 80,712 | 187,624 | -106,912 | -56.98% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-express-route | 40,930 | 40,522 | +408 | +1.01% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-generic-kubernetes | 41,679 | 27,181 | +14,498 | +53.34% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-python-refactor | 40,995 | 40,331 | +664 | +1.65% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-react-optimize | 69,618 | 40,882 | +28,736 | +70.29% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-video-editing | 13,444 | 13,257 | +187 | +1.41% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-cad-to-simready | 67,899 | 13,327 | +54,572 | +409.48% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-dicom | 126,618 | 57,984 | +68,634 | +118.37% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-dynamo-kv | 71,047 | 17,627 | +53,420 | +303.06% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-gpu-pandas | 53,320 | 26,172 | +27,148 | +103.73% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-isaac-lab | 103,549 | 52,434 | +51,115 | +97.48% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-jetson-driver-install | 61,973 | 17,645 | +44,328 | +251.22% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-jetson-setup | 53,437 | 17,828 | +35,609 | +199.74% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-multigpu-training | 78,115 | 66,339 | +11,776 | +17.75% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-openusd-optimization | 54,639 | 65,106 | -10,467 | -16.08% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-vehicle-routing | 70,381 | 38,871 | +31,510 | +81.06% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 4,148,374 | 2,539,548 | +1,608,826 | +63.35% | skill 36/36; base 36/36 |

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

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nvidia-skill-finder/SKILL.md`)
- **MEDIUM** SECURITY/MCP Rug Pull (RP1): MCP Rug Pull: npx commands without a version suffix (e.g. @1.0.0) create a rug-pull risk if the upstream server is compromised and pub (`SKILL.md:74`)
- **MEDIUM** SECURITY/MCP Rug Pull (RP1): MCP Rug Pull: npx commands without a version suffix (e.g. @1.0.0) create a rug-pull risk if the upstream server is compromised and pub (`SKILL.md:103`)
- **MEDIUM** SECURITY/Security finding (SQP-2): Because the skill triggers on many NVIDIA-related requests without being asked, the agent may download and run remote np (`SKILL.md:109`)
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
