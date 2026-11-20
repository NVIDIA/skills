# Skill Benchmark: dicom-series-preflight

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `dicom-series-preflight`
- Validation status: `passed`
- Evaluation date: 2026-11-20
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 3 evaluation tasks (3 positive)
- Dataset digest: `sha256:6a9c1991fda1f2a40a2bc2257509ff06567cbde7ceb769c0946856326899f30f` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| dicom-series-preflight | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| dicom-series-preflight | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| dicom-series-preflight | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| dicom-series-preflight | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| dicom-series-preflight | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `dicom-series-preflight`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 87.0% — baseline ran, but no comparable score was available; uplift unavailable | 88.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 66.7% (-33.3 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 73.3% → 100.0% (+26.7 points) | 46.7% → 100.0% (+53.3 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 81.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 77.8% → 94.5% (+16.7 points) | 36.1% → 83.3% (+47.2 points) |
| Efficiency | 73.8% — baseline ran, but no comparable score was available; uplift unavailable | 79.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 95.5% → 94.4% (-1.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 87.6% → 88.3% (+0.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 888,744 | 500,046 | +388,698 | +77.73% | skill 3/3; base 3/3 |
| claude-code | preflight-clean-axial-flags-phi | 282,154 | 113,318 | +168,836 | +148.99% | skill 1/1; base 1/1 |
| claude-code | preflight-clean-no-phi-greenlights | 324,488 | 153,215 | +171,273 | +111.79% | skill 1/1; base 1/1 |
| claude-code | preflight-flipped-lr-blocks-segmentation | 282,102 | 233,513 | +48,589 | +20.81% | skill 1/1; base 1/1 |
| codex | All cases | 745,033 | 395,784 | +349,249 | +88.24% | skill 3/3; base 3/3 |
| codex | preflight-clean-axial-flags-phi | 219,290 | 105,709 | +113,581 | +107.45% | skill 1/1; base 1/1 |
| codex | preflight-clean-no-phi-greenlights | 274,884 | 195,514 | +79,370 | +40.60% | skill 1/1; base 1/1 |
| codex | preflight-flipped-lr-blocks-segmentation | 250,859 | 94,561 | +156,298 | +165.29% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,633,777 | 895,830 | +737,947 | +82.38% | skill 6/6; base 6/6 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 3 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/dicom-series-preflight/SKILL.md`)
- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution: subprocess.run(
            [sys.executable, str(FIXTURES / "generate_fixtures.py")],
            check=True,
            cwd=REPO,
        ) (`tests/test_preflight_series.py:36`)
- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution: subprocess.run(
        [sys.executable, str(script), "--out-dir", str(destination)], cwd=tmp_path, check=True
    ) (`tests/test_preflight_series.py:69`)
- **LOW** QUALITY/quality_reliability: Inputs are used but no dedicated Inputs section is documented (`skills/dicom-series-preflight/SKILL.md`)
- **LOW** QUALITY/quality_reliability: Structured output is used but no dedicated Output Format section is documented (`skills/dicom-series-preflight/SKILL.md`)
- 7 additional finding(s) are available in the full evaluation artifacts.

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
