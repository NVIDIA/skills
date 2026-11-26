# Skill Benchmark: nim-select

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nim-select`
- Validation status: `passed`
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
| Overall | 87.6% — baseline ran, but no comparable score was available; uplift unavailable | 85.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 91.7% → 97.2% (+5.5 points) | 76.4% → 95.8% (+19.4 points) |
| Correctness | 32.2% → 90.6% (+58.4 points) | 28.9% → 87.8% (+58.9 points) |
| Discoverability | 90.6% — baseline ran, but no comparable score was available; uplift unavailable | 91.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 19.6% → 65.2% (+45.6 points) | 18.6% → 63.9% (+45.3 points) |
| Efficiency | 94.5% — baseline ran, but no comparable score was available; uplift unavailable | 90.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.8% → 91.0% (-6.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 83.6% → 89.2% (+5.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 5,647,718 | 4,749,582 | +898,136 | +18.91% | skill 36/36; base 36/36 |
| claude-code | neg-check-running-nim | 143,101 | 142,491 | +610 | +0.43% | skill 1/1; base 1/1 |
| claude-code | neg-deploy-request | 144,372 | 143,758 | +614 | +0.43% | skill 1/1; base 1/1 |
| claude-code | neg-latency-tuning | 164,123 | 107,752 | +56,371 | +52.32% | skill 1/1; base 1/1 |
| claude-code | real-70b-two-h100 | 142,394 | 36,270 | +106,124 | +292.59% | skill 1/1; base 1/1 |
| claude-code | real-ambiguous-model-name | 142,219 | 72,146 | +70,073 | +97.13% | skill 1/1; base 1/1 |
| claude-code | real-cnd-image-labels | 197,035 | 107,339 | +89,696 | +83.56% | skill 1/1; base 1/1 |
| claude-code | real-cnd-model-name | 196,601 | 107,883 | +88,718 | +82.24% | skill 1/1; base 1/1 |
| claude-code | real-cnd-not-on-that-cnd | 196,527 | 71,817 | +124,710 | +173.65% | skill 1/1; base 1/1 |
| claude-code | real-host-llama-8b | 141,980 | 108,528 | +33,452 | +30.82% | skill 1/1; base 1/1 |
| claude-code | real-image-choice-explained | 142,453 | 413,477 | -271,024 | -65.55% | skill 1/1; base 1/1 |
| claude-code | real-k8s-gpt-oss | 141,856 | 179,268 | -37,412 | -20.87% | skill 1/1; base 1/1 |
| claude-code | real-k8s-selected-gpu-placement-handoff | 181,820 | 222,040 | -40,220 | -18.11% | skill 1/1; base 1/1 |
| claude-code | real-list-cnd | 141,612 | 107,544 | +34,068 | +31.68% | skill 1/1; base 1/1 |
| claude-code | real-list-country-lookup | 142,128 | 109,229 | +32,899 | +30.12% | skill 1/1; base 1/1 |
| claude-code | real-list-earlier-run | 195,810 | 107,021 | +88,789 | +82.96% | skill 1/1; base 1/1 |
| claude-code | real-list-host | 142,054 | 72,343 | +69,711 | +96.36% | skill 1/1; base 1/1 |
| claude-code | real-list-query-k8s | 141,700 | 142,439 | -739 | -0.52% | skill 1/1; base 1/1 |
| claude-code | real-list-region | 141,807 | 186,985 | -45,178 | -24.16% | skill 1/1; base 1/1 |
| claude-code | real-list-then-select | 141,820 | 183,439 | -41,619 | -22.69% | skill 1/1; base 1/1 |
| claude-code | real-locality-ask | 141,781 | 147,633 | -5,852 | -3.96% | skill 1/1; base 1/1 |
| claude-code | real-locality-outside-china | 197,109 | 107,429 | +89,680 | +83.48% | skill 1/1; base 1/1 |
| claude-code | real-locate-then-ask | 142,297 | 143,866 | -1,569 | -1.09% | skill 1/1; base 1/1 |
| claude-code | real-newest-image-does-not-fit | 143,223 | 72,668 | +70,555 | +97.09% | skill 1/1; base 1/1 |
| claude-code | real-pull-to-test-fit | 142,430 | 107,243 | +35,187 | +32.81% | skill 1/1; base 1/1 |
| claude-code | real-select-region | 251,752 | 35,953 | +215,799 | +600.23% | skill 1/1; base 1/1 |
| claude-code | trick-cnd-map-by-name | 197,942 | 182,179 | +15,763 | +8.65% | skill 1/1; base 1/1 |
| claude-code | trick-latest-tag-is-fine | 143,334 | 143,887 | -553 | -0.38% | skill 1/1; base 1/1 |
| claude-code | trick-list-from-memory | 36,459 | 36,574 | -115 | -0.31% | skill 1/1; base 1/1 |
| claude-code | trick-match-by-gpu-name | 144,174 | 143,035 | +1,139 | +0.80% | skill 1/1; base 1/1 |
| claude-code | trick-missing-vram-says-fits | 143,365 | 70,868 | +72,497 | +102.30% | skill 1/1; base 1/1 |
| claude-code | trick-no-metadata-guess | 146,156 | 143,843 | +2,313 | +1.61% | skill 1/1; base 1/1 |
| claude-code | trick-other-api-host | 320,921 | 254,280 | +66,641 | +26.21% | skill 1/1; base 1/1 |
| claude-code | trick-pull-latest-and-try | 142,560 | 106,718 | +35,842 | +33.59% | skill 1/1; base 1/1 |
| claude-code | trick-unknown-is-compatible | 142,530 | 180,747 | -38,217 | -21.14% | skill 1/1; base 1/1 |
| claude-code | trick-unknown-kube-context | 141,939 | 106,793 | +35,146 | +32.91% | skill 1/1; base 1/1 |
| claude-code | trick-use-case-ranking | 88,334 | 144,097 | -55,763 | -38.70% | skill 1/1; base 1/1 |
| codex | All cases | 3,350,755 | 2,700,048 | +650,707 | +24.10% | skill 36/36; base 36/36 |
| codex | neg-check-running-nim | 41,313 | 54,807 | -13,494 | -24.62% | skill 1/1; base 1/1 |
| codex | neg-deploy-request | 42,285 | 56,450 | -14,165 | -25.09% | skill 1/1; base 1/1 |
| codex | neg-latency-tuning | 55,637 | 57,519 | -1,882 | -3.27% | skill 1/1; base 1/1 |
| codex | real-70b-two-h100 | 72,648 | 42,559 | +30,089 | +70.70% | skill 1/1; base 1/1 |
| codex | real-ambiguous-model-name | 119,428 | 35,099 | +84,329 | +240.26% | skill 1/1; base 1/1 |
| codex | real-cnd-image-labels | 140,234 | 112,834 | +27,400 | +24.28% | skill 1/1; base 1/1 |
| codex | real-cnd-model-name | 123,597 | 58,090 | +65,507 | +112.77% | skill 1/1; base 1/1 |
| codex | real-cnd-not-on-that-cnd | 147,867 | 42,818 | +105,049 | +245.34% | skill 1/1; base 1/1 |
| codex | real-host-llama-8b | 76,502 | 237,273 | -160,771 | -67.76% | skill 1/1; base 1/1 |
| codex | real-image-choice-explained | 92,760 | 121,939 | -29,179 | -23.93% | skill 1/1; base 1/1 |
| codex | real-k8s-gpt-oss | 91,778 | 242,511 | -150,733 | -62.16% | skill 1/1; base 1/1 |
| codex | real-k8s-selected-gpu-placement-handoff | 67,944 | 58,445 | +9,499 | +16.25% | skill 1/1; base 1/1 |
| codex | real-list-cnd | 92,224 | 47,441 | +44,783 | +94.40% | skill 1/1; base 1/1 |
| codex | real-list-country-lookup | 76,811 | 128,358 | -51,547 | -40.16% | skill 1/1; base 1/1 |
| codex | real-list-earlier-run | 96,218 | 31,306 | +64,912 | +207.35% | skill 1/1; base 1/1 |
| codex | real-list-host | 96,115 | 155,852 | -59,737 | -38.33% | skill 1/1; base 1/1 |
| codex | real-list-query-k8s | 95,917 | 58,524 | +37,393 | +63.89% | skill 1/1; base 1/1 |
| codex | real-list-region | 100,025 | 65,963 | +34,062 | +51.64% | skill 1/1; base 1/1 |
| codex | real-list-then-select | 72,930 | 107,424 | -34,494 | -32.11% | skill 1/1; base 1/1 |
| codex | real-locality-ask | 89,888 | 51,830 | +38,058 | +73.43% | skill 1/1; base 1/1 |
| codex | real-locality-outside-china | 199,644 | 51,934 | +147,710 | +284.42% | skill 1/1; base 1/1 |
| codex | real-locate-then-ask | 99,922 | 55,734 | +44,188 | +79.28% | skill 1/1; base 1/1 |
| codex | real-newest-image-does-not-fit | 99,910 | 45,789 | +54,121 | +118.20% | skill 1/1; base 1/1 |
| codex | real-pull-to-test-fit | 101,087 | 64,766 | +36,321 | +56.08% | skill 1/1; base 1/1 |
| codex | real-select-region | 149,002 | 47,358 | +101,644 | +214.63% | skill 1/1; base 1/1 |
| codex | trick-cnd-map-by-name | 148,253 | 128,997 | +19,256 | +14.93% | skill 1/1; base 1/1 |
| codex | trick-latest-tag-is-fine | 69,629 | 49,622 | +20,007 | +40.32% | skill 1/1; base 1/1 |
| codex | trick-list-from-memory | 29,976 | 13,903 | +16,073 | +115.61% | skill 1/1; base 1/1 |
| codex | trick-match-by-gpu-name | 95,744 | 40,344 | +55,400 | +137.32% | skill 1/1; base 1/1 |
| codex | trick-missing-vram-says-fits | 98,611 | 51,554 | +47,057 | +91.28% | skill 1/1; base 1/1 |
| codex | trick-no-metadata-guess | 48,956 | 44,707 | +4,249 | +9.50% | skill 1/1; base 1/1 |
| codex | trick-other-api-host | 96,033 | 94,268 | +1,765 | +1.87% | skill 1/1; base 1/1 |
| codex | trick-pull-latest-and-try | 76,762 | 45,426 | +31,336 | +68.98% | skill 1/1; base 1/1 |
| codex | trick-unknown-is-compatible | 96,169 | 49,052 | +47,117 | +96.06% | skill 1/1; base 1/1 |
| codex | trick-unknown-kube-context | 119,066 | 96,613 | +22,453 | +23.24% | skill 1/1; base 1/1 |
| codex | trick-use-case-ranking | 29,870 | 52,939 | -23,069 | -43.58% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 8,998,473 | 7,449,630 | +1,548,843 | +20.79% | skill 72/72; base 72/72 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 70 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 36 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** DUPLICATE/duplicate: Duplicate content found within scripts/ngc_api.py:
  "parse_tag()" in scripts/ngc_api.py (lines 712-717)
  vs "_numbers()" in scripts/ngc_api.py (lines 851-854) (`scripts/ngc_api.py:712`)
- **MEDIUM** PII/ip_addresses: Public IP address (`tests/test_locality.py:207`)
- **MEDIUM** PII/ip_addresses: Public IP address (`tests/test_locality.py:223`)
- **MEDIUM** PII/ip_addresses: Public IP address (`tests/test_locality.py:303`)
- **MEDIUM** PII/ip_addresses: Public IP address (`tests/test_ngc_api.py:342`)
- 66 additional finding(s) are available in the full evaluation artifacts.

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
