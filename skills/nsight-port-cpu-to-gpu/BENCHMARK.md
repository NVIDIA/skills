# Skill Benchmark: nsight-port-cpu-to-gpu

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nsight-port-cpu-to-gpu`
- Validation status: `passed`
- Evaluation date: 2026-11-03
- Evaluator version: `1.6.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-sonnet-5-5`), Codex (`openai/openai/gpt-6-luna`)
- Tasks: 21 evaluation tasks (18 positive, 3 negative)
- Dataset digest: `sha256:97f95413fa7e0482cf071499e79e15bb4c6ff0faf6e060796259828e2200b27c` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nsight-port-cpu-to-gpu | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-sonnet-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| nsight-port-cpu-to-gpu | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-sonnet-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| nsight-port-cpu-to-gpu | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-sonnet-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| nsight-port-cpu-to-gpu | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-sonnet-5-5` | `2.13.3` | nondeterministic / provider_defaults |
| nsight-port-cpu-to-gpu | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-sonnet-5-5` | `2.13.3` | nondeterministic / provider_defaults |

- Controls for `nsight-port-cpu-to-gpu`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 56.5% — baseline ran, but no comparable score was available; uplift unavailable | 77.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 53.3% → 60.0% (+6.7 points) | 46.7% → 74.3% (+27.6 points) |
| Discoverability | 9.9% — baseline ran, but no comparable score was available; uplift unavailable | 75.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 22.9% → 32.3% (+9.4 points) | 22.7% → 55.1% (+32.4 points) |
| Efficiency | 80.4% — baseline ran, but no comparable score was available; uplift unavailable | 84.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.1% → 98.0% (-0.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 91.2% → 90.0% (-1.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,284,777 | 1,083,885 | +200,892 | +18.53% | skill 21/21; base 21/21 |
| claude-code | ag-fault-three-attempts | 36,341 | 36,004 | +337 | +0.94% | skill 1/1; base 1/1 |
| claude-code | ag-gate-low-engagement | 35,989 | 35,856 | +133 | +0.37% | skill 1/1; base 1/1 |
| claude-code | ag-gpu-marker-missing | 35,786 | 35,552 | +234 | +0.66% | skill 1/1; base 1/1 |
| claude-code | ag-injection-build-log | 36,339 | 36,136 | +203 | +0.56% | skill 1/1; base 1/1 |
| claude-code | ag-injection-source-comment | 35,894 | 35,568 | +326 | +0.92% | skill 1/1; base 1/1 |
| claude-code | ag-negative-cpu-refactor | 150,575 | 105,693 | +44,882 | +42.46% | skill 1/1; base 1/1 |
| claude-code | ag-negative-cuda-crash | 36,026 | 35,943 | +83 | +0.23% | skill 1/1; base 1/1 |
| claude-code | ag-negative-pytorch | 35,947 | 35,824 | +123 | +0.34% | skill 1/1; base 1/1 |
| claude-code | ag-plan-from-profile | 148,484 | 186,385 | -37,901 | -20.33% | skill 1/1; base 1/1 |
| claude-code | ag-plan-from-trace | 36,785 | 36,404 | +381 | +1.05% | skill 1/1; base 1/1 |
| claude-code | ag-resume-in-progress | 35,600 | 35,339 | +261 | +0.74% | skill 1/1; base 1/1 |
| claude-code | ag-route-cpp-no-fortran-target | 35,668 | 35,680 | -12 | -0.03% | skill 1/1; base 1/1 |
| claude-code | ag-route-cross-language | 36,822 | 36,685 | +137 | +0.37% | skill 1/1; base 1/1 |
| claude-code | ag-route-flavor-openacc | 188,064 | 35,767 | +152,297 | +425.80% | skill 1/1; base 1/1 |
| claude-code | ag-setup-start-port | 185,360 | 147,020 | +38,340 | +26.08% | skill 1/1; base 1/1 |
| claude-code | ag-translate-plan-order | 35,713 | 35,364 | +349 | +0.99% | skill 1/1; base 1/1 |
| claude-code | ag-trap-cpu-gpu-agree | 35,758 | 35,770 | -12 | -0.03% | skill 1/1; base 1/1 |
| claude-code | ag-trap-global-integral | 36,037 | 35,653 | +384 | +1.08% | skill 1/1; base 1/1 |
| claude-code | ag-trap-skip-standalone-test | 35,655 | 35,446 | +209 | +0.59% | skill 1/1; base 1/1 |
| claude-code | ag-verify-outlier-boundary | 35,948 | 35,770 | +178 | +0.50% | skill 1/1; base 1/1 |
| claude-code | ag-verify-test-main-language | 35,986 | 36,026 | -40 | -0.11% | skill 1/1; base 1/1 |
| codex | All cases | 1,435,741 | 944,703 | +491,038 | +51.98% | skill 21/21; base 21/21 |
| codex | ag-fault-three-attempts | 30,755 | 18,538 | +12,217 | +65.90% | skill 1/1; base 1/1 |
| codex | ag-gate-low-engagement | 67,248 | 13,561 | +53,687 | +395.89% | skill 1/1; base 1/1 |
| codex | ag-gpu-marker-missing | 94,216 | 28,963 | +65,253 | +225.30% | skill 1/1; base 1/1 |
| codex | ag-injection-build-log | 14,219 | 13,623 | +596 | +4.37% | skill 1/1; base 1/1 |
| codex | ag-injection-source-comment | 30,322 | 13,320 | +17,002 | +127.64% | skill 1/1; base 1/1 |
| codex | ag-negative-cpu-refactor | 71,317 | 73,807 | -2,490 | -3.37% | skill 1/1; base 1/1 |
| codex | ag-negative-cuda-crash | 22,048 | 22,372 | -324 | -1.45% | skill 1/1; base 1/1 |
| codex | ag-negative-pytorch | 22,207 | 22,405 | -198 | -0.88% | skill 1/1; base 1/1 |
| codex | ag-plan-from-profile | 158,809 | 195,421 | -36,612 | -18.73% | skill 1/1; base 1/1 |
| codex | ag-plan-from-trace | 119,214 | 22,747 | +96,467 | +424.09% | skill 1/1; base 1/1 |
| codex | ag-resume-in-progress | 124,722 | 13,386 | +111,336 | +831.73% | skill 1/1; base 1/1 |
| codex | ag-route-cpp-no-fortran-target | 64,457 | 13,448 | +51,009 | +379.31% | skill 1/1; base 1/1 |
| codex | ag-route-cross-language | 90,936 | 31,112 | +59,824 | +192.29% | skill 1/1; base 1/1 |
| codex | ag-route-flavor-openacc | 119,041 | 14,207 | +104,834 | +737.90% | skill 1/1; base 1/1 |
| codex | ag-setup-start-port | 115,498 | 303,739 | -188,241 | -61.97% | skill 1/1; base 1/1 |
| codex | ag-translate-plan-order | 83,875 | 13,292 | +70,583 | +531.02% | skill 1/1; base 1/1 |
| codex | ag-trap-cpu-gpu-agree | 18,134 | 13,434 | +4,700 | +34.99% | skill 1/1; base 1/1 |
| codex | ag-trap-global-integral | 72,796 | 72,374 | +422 | +0.58% | skill 1/1; base 1/1 |
| codex | ag-trap-skip-standalone-test | 67,790 | 13,325 | +54,465 | +408.74% | skill 1/1; base 1/1 |
| codex | ag-verify-outlier-boundary | 23,342 | 18,078 | +5,264 | +29.12% | skill 1/1; base 1/1 |
| codex | ag-verify-test-main-language | 24,795 | 13,551 | +11,244 | +82.98% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,720,518 | 2,028,588 | +691,930 | +34.11% | skill 42/42; base 42/42 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 24 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 69 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 21 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across flavors/cpp/cuda/translation.md and flavors/cpp/kokkos/translation.md and flavors/fortran/openacc/translation-sibling.md and flavors/fortran/openmp/translation-sibling.md:
  "## Phase 4b: Scaffold Test Harness" in flavors/cpp/cuda/translation.md (lines 95-100)
  vs "## Phase 4b: Scaffold Test Harness" in flavors/cpp/kokkos/translation.md (lines 90-95)
  vs "## Phase 4b: Scaffold Test Harness" in flavors/fortran/openacc/translation-sibling.md (lines 81-86)
  vs "## Phase 4b: Scaffold Test Harness" in flavors/fortran/openmp/translation-sibling.md (lines 88-93) (`flavors/cpp/cuda/translation.md:95`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across flavors/cpp/cuda/translation.md and flavors/cpp/kokkos/translation.md and flavors/fortran/openacc/translation-inline.md and flavors/fortran/openacc/translation-sibling.md and flavors/fortran/openmp/translation-inline.md and flavors/fortran/openmp/translation-sibling.md and orchestrators/nsys/references/planning-from-trace.md and transitions/fortran-to-cpp.md:
  "### Prerequisite: function must be externally callable" in flavors/cpp/cuda/translation.md (lines 101-112)
  vs "### Prerequisite: function must be externally callable" in flavors/cpp/kokkos/translation.md (lines 96-108)
  vs "## Phase 4b: Scaffold (reference copy + test)" in flavors/fortran/openacc/translation-inline.md (lines 61-69)
  vs "### Prerequisite: function must be externally callable" in flavors/fortran/openacc/translation-sibling.md (lines 87-100)
  vs "## Phase 4b: Scaffold (reference copy + test)" in flavors/fortran/openmp/translation-inline.md (lines 65-73)
  vs "### Prerequisite: function must be externally callable" in flavors/fortran/openmp/translation-sibling.md (lines 94-107)
  vs "## Callability Check (MANDATORY)" in orchestrators/nsys/references/planning-from-trace.md (lines 176-208)
  vs "#### Prerequisite: externally callable" in transitions/fortran-to-cpp.md (lines 157-164) (`flavors/cpp/cuda/translation.md:101`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across flavors/cpp/cuda/translation.md and flavors/cpp/kokkos/translation.md and flavors/fortran/openacc/translation-sibling.md and flavors/fortran/openmp/translation-sibling.md:
  "### Steps" in flavors/cpp/cuda/translation.md (lines 157-164)
  vs "### Steps" in flavors/cpp/kokkos/translation.md (lines 157-164)
  vs "### Steps" in flavors/fortran/openacc/translation-sibling.md (lines 132-136)
  vs "### Steps" in flavors/fortran/openmp/translation-sibling.md (lines 139-143) (`flavors/cpp/cuda/translation.md:157`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across flavors/cpp/cuda/translation.md and flavors/cpp/kokkos/translation.md and flavors/fortran/openacc/translation-sibling.md and flavors/fortran/openmp/translation-sibling.md:
  "### Steps" in flavors/cpp/cuda/translation.md (lines 240-243)
  vs "### Steps" in flavors/cpp/kokkos/translation.md (lines 242-246)
  vs "### Steps" in flavors/fortran/openacc/translation-sibling.md (lines 200-204)
  vs "### Steps" in flavors/fortran/openmp/translation-sibling.md (lines 207-211) (`flavors/cpp/cuda/translation.md:240`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across flavors/cpp/cuda/translation.md and flavors/cpp/kokkos/translation.md and flavors/fortran/openacc/translation-sibling.md and flavors/fortran/openmp/translation-sibling.md:
  "### Steps" in flavors/cpp/cuda/translation.md (lines 144-154)
  vs "### Steps" in flavors/cpp/kokkos/translation.md (lines 145-154)
  vs "### Steps" in flavors/fortran/openacc/translation-sibling.md (lines 119-126)
  vs "### Steps" in flavors/fortran/openmp/translation-sibling.md (lines 126-133) (`flavors/cpp/cuda/translation.md:144`)
- 88 additional finding(s) are available in the full evaluation artifacts.

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
