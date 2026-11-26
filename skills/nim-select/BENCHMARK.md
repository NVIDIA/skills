# Skill Benchmark: nim-select

> ⚠️ **Overall verdict: INCOMPLETE — Required evidence is missing**

One or more required evaluation tiers did not complete, so this benchmark is not publication-complete.

## Evaluation Metadata

- Skill: `nim-select`
- Validation status: `incomplete`
- Evaluation date: 2026-11-26
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 36 evaluation tasks (36 positive)
- Dataset digest: `sha256:56820f16a7580a22f2ebc2adb6773432ffded122b6fcec60da496513b75dd797` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nim-select | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nim-select | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nim-select | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nim-select | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nim-select | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `nim-select`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 88.0% — baseline ran, but no comparable score was available; uplift unavailable | 85.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 93.1% → 98.6% (+5.5 points) | 81.9% → 95.8% (+13.9 points) |
| Correctness | 33.3% → 92.2% (+58.9 points) | 31.1% → 89.4% (+58.3 points) |
| Discoverability | 91.8% — baseline ran, but no comparable score was available; uplift unavailable | 90.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 20.4% → 62.8% (+42.4 points) | 19.0% → 61.5% (+42.5 points) |
| Efficiency | 94.5% — baseline ran, but no comparable score was available; uplift unavailable | 90.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.1% → 91.3% (-6.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 83.7% → 89.2% (+5.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 5,650,980 | 4,401,130 | +1,249,850 | +28.40% | skill 36/36; base 36/36 |
| claude-code | neg-check-running-nim | 143,424 | 142,382 | +1,042 | +0.73% | skill 1/1; base 1/1 |
| claude-code | neg-deploy-request | 109,389 | 181,069 | -71,680 | -39.59% | skill 1/1; base 1/1 |
| claude-code | neg-latency-tuning | 191,450 | 143,952 | +47,498 | +33.00% | skill 1/1; base 1/1 |
| claude-code | real-70b-two-h100 | 142,251 | 36,243 | +106,008 | +292.49% | skill 1/1; base 1/1 |
| claude-code | real-ambiguous-model-name | 144,258 | 72,087 | +72,171 | +100.12% | skill 1/1; base 1/1 |
| claude-code | real-cnd-image-labels | 197,212 | 144,031 | +53,181 | +36.92% | skill 1/1; base 1/1 |
| claude-code | real-cnd-model-name | 196,390 | 108,041 | +88,349 | +81.77% | skill 1/1; base 1/1 |
| claude-code | real-cnd-not-on-that-cnd | 196,654 | 70,915 | +125,739 | +177.31% | skill 1/1; base 1/1 |
| claude-code | real-host-llama-8b | 141,925 | 108,464 | +33,461 | +30.85% | skill 1/1; base 1/1 |
| claude-code | real-image-choice-explained | 142,639 | 182,169 | -39,530 | -21.70% | skill 1/1; base 1/1 |
| claude-code | real-k8s-gpt-oss | 142,032 | 142,577 | -545 | -0.38% | skill 1/1; base 1/1 |
| claude-code | real-k8s-selected-gpu-placement-handoff | 197,867 | 143,258 | +54,609 | +38.12% | skill 1/1; base 1/1 |
| claude-code | real-list-cnd | 141,637 | 72,130 | +69,507 | +96.36% | skill 1/1; base 1/1 |
| claude-code | real-list-country-lookup | 142,032 | 182,764 | -40,732 | -22.29% | skill 1/1; base 1/1 |
| claude-code | real-list-earlier-run | 195,825 | 216,616 | -20,791 | -9.60% | skill 1/1; base 1/1 |
| claude-code | real-list-host | 141,878 | 108,777 | +33,101 | +30.43% | skill 1/1; base 1/1 |
| claude-code | real-list-query-k8s | 141,960 | 143,662 | -1,702 | -1.18% | skill 1/1; base 1/1 |
| claude-code | real-list-region | 141,594 | 147,151 | -5,557 | -3.78% | skill 1/1; base 1/1 |
| claude-code | real-list-then-select | 141,841 | 184,924 | -43,083 | -23.30% | skill 1/1; base 1/1 |
| claude-code | real-locality-ask | 141,846 | 107,622 | +34,224 | +31.80% | skill 1/1; base 1/1 |
| claude-code | real-locality-outside-china | 252,111 | 143,655 | +108,456 | +75.50% | skill 1/1; base 1/1 |
| claude-code | real-locate-then-ask | 142,275 | 107,982 | +34,293 | +31.76% | skill 1/1; base 1/1 |
| claude-code | real-newest-image-does-not-fit | 142,404 | 72,333 | +70,071 | +96.87% | skill 1/1; base 1/1 |
| claude-code | real-pull-to-test-fit | 142,466 | 107,066 | +35,400 | +33.06% | skill 1/1; base 1/1 |
| claude-code | real-select-region | 251,898 | 36,107 | +215,791 | +597.64% | skill 1/1; base 1/1 |
| claude-code | trick-cnd-map-by-name | 197,753 | 181,431 | +16,322 | +9.00% | skill 1/1; base 1/1 |
| claude-code | trick-latest-tag-is-fine | 143,443 | 107,523 | +35,920 | +33.41% | skill 1/1; base 1/1 |
| claude-code | trick-list-from-memory | 36,633 | 36,477 | +156 | +0.43% | skill 1/1; base 1/1 |
| claude-code | trick-match-by-gpu-name | 144,100 | 179,471 | -35,371 | -19.71% | skill 1/1; base 1/1 |
| claude-code | trick-missing-vram-says-fits | 143,347 | 70,894 | +72,453 | +102.20% | skill 1/1; base 1/1 |
| claude-code | trick-no-metadata-guess | 142,734 | 143,883 | -1,149 | -0.80% | skill 1/1; base 1/1 |
| claude-code | trick-other-api-host | 262,000 | 217,594 | +44,406 | +20.41% | skill 1/1; base 1/1 |
| claude-code | trick-pull-latest-and-try | 142,929 | 71,052 | +71,877 | +101.16% | skill 1/1; base 1/1 |
| claude-code | trick-unknown-is-compatible | 142,368 | 106,940 | +35,428 | +33.13% | skill 1/1; base 1/1 |
| claude-code | trick-unknown-kube-context | 141,971 | 107,392 | +34,579 | +32.20% | skill 1/1; base 1/1 |
| claude-code | trick-use-case-ranking | 88,444 | 72,496 | +15,948 | +22.00% | skill 1/1; base 1/1 |
| codex | All cases | 3,307,248 | 2,537,081 | +770,167 | +30.36% | skill 36/36; base 36/36 |
| codex | neg-check-running-nim | 55,598 | 40,505 | +15,093 | +37.26% | skill 1/1; base 1/1 |
| codex | neg-deploy-request | 42,204 | 42,327 | -123 | -0.29% | skill 1/1; base 1/1 |
| codex | neg-latency-tuning | 51,404 | 80,666 | -29,262 | -36.28% | skill 1/1; base 1/1 |
| codex | real-70b-two-h100 | 99,923 | 47,079 | +52,844 | +112.25% | skill 1/1; base 1/1 |
| codex | real-ambiguous-model-name | 95,516 | 35,094 | +60,422 | +172.17% | skill 1/1; base 1/1 |
| codex | real-cnd-image-labels | 93,859 | 74,362 | +19,497 | +26.22% | skill 1/1; base 1/1 |
| codex | real-cnd-model-name | 118,552 | 66,393 | +52,159 | +78.56% | skill 1/1; base 1/1 |
| codex | real-cnd-not-on-that-cnd | 95,209 | 61,622 | +33,587 | +54.50% | skill 1/1; base 1/1 |
| codex | real-host-llama-8b | 95,710 | 79,085 | +16,625 | +21.02% | skill 1/1; base 1/1 |
| codex | real-image-choice-explained | 95,779 | 132,528 | -36,749 | -27.73% | skill 1/1; base 1/1 |
| codex | real-k8s-gpt-oss | 95,554 | 301,630 | -206,076 | -68.32% | skill 1/1; base 1/1 |
| codex | real-k8s-selected-gpu-placement-handoff | 117,103 | 58,364 | +58,739 | +100.64% | skill 1/1; base 1/1 |
| codex | real-list-cnd | 75,232 | 57,041 | +18,191 | +31.89% | skill 1/1; base 1/1 |
| codex | real-list-country-lookup | 126,146 | 86,897 | +39,249 | +45.17% | skill 1/1; base 1/1 |
| codex | real-list-earlier-run | 95,845 | 13,306 | +82,539 | +620.31% | skill 1/1; base 1/1 |
| codex | real-list-host | 72,879 | 74,742 | -1,863 | -2.49% | skill 1/1; base 1/1 |
| codex | real-list-query-k8s | 119,426 | 74,980 | +44,446 | +59.28% | skill 1/1; base 1/1 |
| codex | real-list-region | 99,911 | 73,417 | +26,494 | +36.09% | skill 1/1; base 1/1 |
| codex | real-list-then-select | 96,092 | 59,992 | +36,100 | +60.17% | skill 1/1; base 1/1 |
| codex | real-locality-ask | 89,670 | 52,766 | +36,904 | +69.94% | skill 1/1; base 1/1 |
| codex | real-locality-outside-china | 149,041 | 60,178 | +88,863 | +147.67% | skill 1/1; base 1/1 |
| codex | real-locate-then-ask | 100,099 | 72,066 | +28,033 | +38.90% | skill 1/1; base 1/1 |
| codex | real-newest-image-does-not-fit | 99,972 | 46,830 | +53,142 | +113.48% | skill 1/1; base 1/1 |
| codex | real-pull-to-test-fit | 124,019 | 69,414 | +54,605 | +78.67% | skill 1/1; base 1/1 |
| codex | real-select-region | 167,520 | 47,318 | +120,202 | +254.03% | skill 1/1; base 1/1 |
| codex | trick-cnd-map-by-name | 116,206 | 78,671 | +37,535 | +47.71% | skill 1/1; base 1/1 |
| codex | trick-latest-tag-is-fine | 72,583 | 47,500 | +25,083 | +52.81% | skill 1/1; base 1/1 |
| codex | trick-list-from-memory | 34,228 | 13,823 | +20,405 | +147.62% | skill 1/1; base 1/1 |
| codex | trick-match-by-gpu-name | 99,878 | 40,533 | +59,345 | +146.41% | skill 1/1; base 1/1 |
| codex | trick-missing-vram-says-fits | 75,125 | 39,342 | +35,783 | +90.95% | skill 1/1; base 1/1 |
| codex | trick-no-metadata-guess | 48,936 | 53,682 | -4,746 | -8.84% | skill 1/1; base 1/1 |
| codex | trick-other-api-host | 119,367 | 98,427 | +20,940 | +21.27% | skill 1/1; base 1/1 |
| codex | trick-pull-latest-and-try | 70,315 | 48,684 | +21,631 | +44.43% | skill 1/1; base 1/1 |
| codex | trick-unknown-is-compatible | 72,929 | 39,578 | +33,351 | +84.27% | skill 1/1; base 1/1 |
| codex | trick-unknown-kube-context | 95,564 | 224,092 | -128,528 | -57.36% | skill 1/1; base 1/1 |
| codex | trick-use-case-ranking | 29,854 | 44,147 | -14,293 | -32.38% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 8,958,228 | 6,938,211 | +2,020,017 | +29.11% | skill 72/72; base 72/72 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 70 finding(s) |
| Tier 2 | Semantic deduplication | **INCOMPLETE** | Tier 2 scan incomplete: a content cluster has more than 64 members. |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 36 task(s) |

## Incomplete Coverage

- **Context Deduplication:** Tier 2 scan incomplete: a content cluster has more than 64 members.

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** PII/ip_addresses: Public IP address (`tests/test_locality.py:207`)
- **MEDIUM** PII/ip_addresses: Public IP address (`tests/test_locality.py:223`)
- **MEDIUM** PII/ip_addresses: Public IP address (`tests/test_locality.py:303`)
- **MEDIUM** PII/ip_addresses: Public IP address (`tests/test_ngc_api.py:342`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/nim-select/SKILL.md`)
- 65 additional finding(s) are available in the full evaluation artifacts.

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
