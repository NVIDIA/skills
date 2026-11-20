# Skill Benchmark: nv-reason-ct

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nv-reason-ct`
- Validation status: `passed`
- Evaluation date: 2026-11-20
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 9 evaluation tasks (9 positive)
- Dataset digest: `sha256:df577b70bec14f79cd9d434a2db86fd8aaab1d9fda79314f3d42d6e9494e6ab9` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nv-reason-ct | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nv-reason-ct | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nv-reason-ct | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nv-reason-ct | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nv-reason-ct | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `nv-reason-ct`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 91.8% — baseline ran, but no comparable score was available; uplift unavailable | 93.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 94.4% → 100.0% (+5.6 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 53.3% → 100.0% (+46.7 points) | 57.8% → 95.6% (+37.8 points) |
| Discoverability | 85.0% — baseline ran, but no comparable score was available; uplift unavailable | 92.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 55.2% → 85.7% (+30.5 points) | 47.8% → 89.8% (+42.0 points) |
| Efficiency | 88.0% — baseline ran, but no comparable score was available; uplift unavailable | 88.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.9% → 94.2% (-3.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 91.3% → 89.8% (-1.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,520,619 | 1,268,255 | +252,364 | +19.90% | skill 9/9; base 9/9 |
| claude-code | diagnosis-treatment-refusal | 35,335 | 107,199 | -71,864 | -67.04% | skill 1/1; base 1/1 |
| claude-code | git-lfs-pointer-is-not-volume | 170,443 | 107,383 | +63,060 | +58.72% | skill 1/1; base 1/1 |
| claude-code | live-abdominal-nifti-routing | 264,477 | 144,316 | +120,161 | +83.26% | skill 1/1; base 1/1 |
| claude-code | local-trained-export-routing | 171,565 | 144,301 | +27,264 | +18.89% | skill 1/1; base 1/1 |
| claude-code | model-code-consent-withheld | 206,922 | 145,927 | +60,995 | +41.80% | skill 1/1; base 1/1 |
| claude-code | no-silent-history-loss | 151,971 | 106,690 | +45,281 | +42.44% | skill 1/1; base 1/1 |
| claude-code | read-only-setup-check | 221,074 | 185,310 | +35,764 | +19.30% | skill 1/1; base 1/1 |
| claude-code | training-records-are-not-demo-volumes | 77,636 | 181,910 | -104,274 | -57.32% | skill 1/1; base 1/1 |
| claude-code | upstream-example-end-to-end | 221,196 | 145,219 | +75,977 | +52.32% | skill 1/1; base 1/1 |
| codex | All cases | 835,707 | 532,679 | +303,028 | +56.89% | skill 9/9; base 9/9 |
| codex | diagnosis-treatment-refusal | 29,313 | 17,414 | +11,899 | +68.33% | skill 1/1; base 1/1 |
| codex | git-lfs-pointer-is-not-volume | 67,736 | 59,069 | +8,667 | +14.67% | skill 1/1; base 1/1 |
| codex | live-abdominal-nifti-routing | 112,880 | 62,965 | +49,915 | +79.27% | skill 1/1; base 1/1 |
| codex | local-trained-export-routing | 138,572 | 55,508 | +83,064 | +149.64% | skill 1/1; base 1/1 |
| codex | model-code-consent-withheld | 70,196 | 45,694 | +24,502 | +53.62% | skill 1/1; base 1/1 |
| codex | no-silent-history-loss | 47,768 | 13,217 | +34,551 | +261.41% | skill 1/1; base 1/1 |
| codex | read-only-setup-check | 143,666 | 106,235 | +37,431 | +35.23% | skill 1/1; base 1/1 |
| codex | training-records-are-not-demo-volumes | 56,776 | 60,056 | -3,280 | -5.46% | skill 1/1; base 1/1 |
| codex | upstream-example-end-to-end | 168,800 | 112,521 | +56,279 | +50.02% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,356,326 | 1,800,934 | +555,392 | +30.84% | skill 18/18; base 18/18 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 13 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 9 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution: subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=30,
    ) (`tests/test_nv_reason_ct.py:100`)
- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution: subprocess.run(
        command,
        cwd=output_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    ) (`tests/test_upstream_examples.py:74`)
- **MEDIUM** SECURITY/Tainted flow: 'env' from os.environ.get (line 70, credential/environment) → subprocess.run (code execution) (TT2): Data Flow: subprocess.run(
        command,
        cwd=output_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    ) (`tests/test_upstream_examples.py:74`)
- **LOW** QUALITY/quality_reliability: Inputs are used but no dedicated Inputs section is documented (`skills/nv-reason-ct/SKILL.md`)
- **LOW** QUALITY/quality_reliability: Structured output is used but no dedicated Output Format section is documented (`skills/nv-reason-ct/SKILL.md`)
- 8 additional finding(s) are available in the full evaluation artifacts.

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
