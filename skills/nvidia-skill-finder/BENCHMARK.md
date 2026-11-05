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
| Overall | 93.8% — baseline ran, but no comparable score was available; uplift unavailable | 87.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 97.2% (-2.8 points) | 94.4% → 86.1% (-8.3 points) |
| Correctness | 65.6% → 96.7% (+31.1 points) | 76.7% → 95.6% (+18.9 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 35.1% → 96.5% (+61.4 points) | 61.6% → 97.2% (+35.6 points) |
| Efficiency | 78.7% — baseline ran, but no comparable score was available; uplift unavailable | 63.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.3% → 96.3% (-2.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 89.5% → 88.3% (-1.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,756,252 | 1,731,091 | +1,025,161 | +59.22% | skill 18/18; base 18/18 |
| claude-code | nvidia-skill-finder-diff-bare-invocation | 158,059 | 107,030 | +51,029 | +47.68% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-diff-browse-no-keywords | 164,344 | 70,831 | +93,513 | +132.02% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-diff-deep-research | 194,159 | 35,239 | +158,920 | +450.98% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-express-route | 187,002 | 221,909 | -34,907 | -15.73% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-generic-kubernetes | 108,498 | 143,628 | -35,130 | -24.46% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-python-refactor | 107,980 | 141,827 | -33,847 | -23.86% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-react-optimize | 108,155 | 106,957 | +1,198 | +1.12% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-neg-video-editing | 108,223 | 107,431 | +792 | +0.74% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-cad-to-simready | 153,854 | 106,990 | +46,864 | +43.80% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-dicom | 153,851 | 35,602 | +118,249 | +332.14% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-dynamo-kv | 234,195 | 143,882 | +90,313 | +62.77% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-gpu-pandas | 153,696 | 107,250 | +46,446 | +43.31% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-isaac-lab | 154,768 | 37,091 | +117,677 | +317.27% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-jetson-driver-install | 154,068 | 35,956 | +118,112 | +328.49% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-jetson-setup | 153,631 | 107,202 | +46,429 | +43.31% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-multigpu-training | 154,932 | 151,307 | +3,625 | +2.40% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-openusd-optimization | 153,236 | 35,425 | +117,811 | +332.56% | skill 1/1; base 1/1 |
| claude-code | nvidia-skill-finder-pos-vehicle-routing | 153,601 | 35,534 | +118,067 | +332.26% | skill 1/1; base 1/1 |
| codex | All cases | 1,243,207 | 702,854 | +540,353 | +76.88% | skill 18/18; base 18/18 |
| codex | nvidia-skill-finder-diff-bare-invocation | 65,133 | 39,985 | +25,148 | +62.89% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-diff-browse-no-keywords | 78,147 | 48,458 | +29,689 | +61.27% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-diff-deep-research | 102,367 | 48,821 | +53,546 | +109.68% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-express-route | 41,069 | 40,479 | +590 | +1.46% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-generic-kubernetes | 41,698 | 41,283 | +415 | +1.01% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-python-refactor | 41,350 | 40,372 | +978 | +2.42% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-react-optimize | 41,009 | 82,437 | -41,428 | -50.25% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-neg-video-editing | 40,857 | 13,264 | +27,593 | +208.03% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-cad-to-simready | 67,947 | 13,334 | +54,613 | +409.58% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-dicom | 74,682 | 30,159 | +44,523 | +147.63% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-dynamo-kv | 71,009 | 45,191 | +25,818 | +57.13% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-gpu-pandas | 68,996 | 21,810 | +47,186 | +216.35% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-isaac-lab | 146,943 | 67,402 | +79,541 | +118.01% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-jetson-driver-install | 73,345 | 17,678 | +55,667 | +314.89% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-jetson-setup | 53,502 | 17,751 | +35,751 | +201.40% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-multigpu-training | 57,962 | 30,505 | +27,457 | +90.01% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-openusd-optimization | 68,939 | 67,955 | +984 | +1.45% | skill 1/1; base 1/1 |
| codex | nvidia-skill-finder-pos-vehicle-routing | 108,252 | 35,970 | +72,282 | +200.95% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 3,999,459 | 2,433,945 | +1,565,514 | +64.32% | skill 36/36; base 36/36 |

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

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nvidia-skill-finder/SKILL.md`)
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
