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
| Overall | 87.9% — baseline ran, but no comparable score was available; uplift unavailable | 85.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 93.1% → 98.6% (+5.5 points) | 83.3% → 95.8% (+12.5 points) |
| Correctness | 34.4% → 90.6% (+56.2 points) | 28.9% → 91.7% (+62.8 points) |
| Discoverability | 89.8% — baseline ran, but no comparable score was available; uplift unavailable | 89.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 19.8% → 65.9% (+46.1 points) | 19.2% → 59.9% (+40.7 points) |
| Efficiency | 94.5% — baseline ran, but no comparable score was available; uplift unavailable | 89.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.8% → 91.3% (-6.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 84.1% → 89.1% (+5.0 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 5,459,656 | 4,645,430 | +814,226 | +17.53% | skill 36/36; base 36/36 |
| claude-code | neg-check-running-nim | 142,982 | 146,732 | -3,750 | -2.56% | skill 1/1; base 1/1 |
| claude-code | neg-deploy-request | 107,769 | 217,816 | -110,047 | -50.52% | skill 1/1; base 1/1 |
| claude-code | neg-latency-tuning | 180,956 | 145,559 | +35,397 | +24.32% | skill 1/1; base 1/1 |
| claude-code | real-70b-two-h100 | 142,971 | 71,873 | +71,098 | +98.92% | skill 1/1; base 1/1 |
| claude-code | real-ambiguous-model-name | 142,209 | 72,436 | +69,773 | +96.32% | skill 1/1; base 1/1 |
| claude-code | real-cnd-image-labels | 197,061 | 107,660 | +89,401 | +83.04% | skill 1/1; base 1/1 |
| claude-code | real-cnd-model-name | 196,318 | 145,212 | +51,106 | +35.19% | skill 1/1; base 1/1 |
| claude-code | real-cnd-not-on-that-cnd | 196,329 | 71,463 | +124,866 | +174.73% | skill 1/1; base 1/1 |
| claude-code | real-host-llama-8b | 141,959 | 184,890 | -42,931 | -23.22% | skill 1/1; base 1/1 |
| claude-code | real-image-choice-explained | 141,995 | 182,106 | -40,111 | -22.03% | skill 1/1; base 1/1 |
| claude-code | real-k8s-gpt-oss | 142,043 | 178,556 | -36,513 | -20.45% | skill 1/1; base 1/1 |
| claude-code | real-k8s-selected-gpu-placement-handoff | 187,820 | 180,810 | +7,010 | +3.88% | skill 1/1; base 1/1 |
| claude-code | real-list-cnd | 141,656 | 108,080 | +33,576 | +31.07% | skill 1/1; base 1/1 |
| claude-code | real-list-country-lookup | 141,963 | 187,150 | -45,187 | -24.14% | skill 1/1; base 1/1 |
| claude-code | real-list-earlier-run | 195,579 | 142,746 | +52,833 | +37.01% | skill 1/1; base 1/1 |
| claude-code | real-list-host | 141,878 | 108,613 | +33,265 | +30.63% | skill 1/1; base 1/1 |
| claude-code | real-list-query-k8s | 141,701 | 142,524 | -823 | -0.58% | skill 1/1; base 1/1 |
| claude-code | real-list-region | 141,805 | 108,896 | +32,909 | +30.22% | skill 1/1; base 1/1 |
| claude-code | real-list-then-select | 141,938 | 185,854 | -43,916 | -23.63% | skill 1/1; base 1/1 |
| claude-code | real-locality-ask | 141,859 | 108,064 | +33,795 | +31.27% | skill 1/1; base 1/1 |
| claude-code | real-locality-outside-china | 196,699 | 145,957 | +50,742 | +34.77% | skill 1/1; base 1/1 |
| claude-code | real-locate-then-ask | 142,195 | 187,494 | -45,299 | -24.16% | skill 1/1; base 1/1 |
| claude-code | real-newest-image-does-not-fit | 142,512 | 36,469 | +106,043 | +290.78% | skill 1/1; base 1/1 |
| claude-code | real-pull-to-test-fit | 142,562 | 107,457 | +35,105 | +32.67% | skill 1/1; base 1/1 |
| claude-code | real-select-region | 142,179 | 36,456 | +105,723 | +290.00% | skill 1/1; base 1/1 |
| claude-code | trick-cnd-map-by-name | 198,053 | 145,614 | +52,439 | +36.01% | skill 1/1; base 1/1 |
| claude-code | trick-latest-tag-is-fine | 143,011 | 143,375 | -364 | -0.25% | skill 1/1; base 1/1 |
| claude-code | trick-list-from-memory | 36,537 | 36,346 | +191 | +0.53% | skill 1/1; base 1/1 |
| claude-code | trick-match-by-gpu-name | 143,492 | 143,203 | +289 | +0.20% | skill 1/1; base 1/1 |
| claude-code | trick-missing-vram-says-fits | 143,342 | 71,308 | +72,034 | +101.02% | skill 1/1; base 1/1 |
| claude-code | trick-no-metadata-guess | 142,469 | 108,419 | +34,050 | +31.41% | skill 1/1; base 1/1 |
| claude-code | trick-other-api-host | 262,199 | 255,459 | +6,740 | +2.64% | skill 1/1; base 1/1 |
| claude-code | trick-pull-latest-and-try | 142,568 | 71,056 | +71,512 | +100.64% | skill 1/1; base 1/1 |
| claude-code | trick-unknown-is-compatible | 142,578 | 107,038 | +35,540 | +33.20% | skill 1/1; base 1/1 |
| claude-code | trick-unknown-kube-context | 142,066 | 107,632 | +34,434 | +31.99% | skill 1/1; base 1/1 |
| claude-code | trick-use-case-ranking | 88,403 | 145,107 | -56,704 | -39.08% | skill 1/1; base 1/1 |
| codex | All cases | 3,607,139 | 2,454,270 | +1,152,869 | +46.97% | skill 36/36; base 36/36 |
| codex | neg-check-running-nim | 40,811 | 85,131 | -44,320 | -52.06% | skill 1/1; base 1/1 |
| codex | neg-deploy-request | 42,124 | 55,786 | -13,662 | -24.49% | skill 1/1; base 1/1 |
| codex | neg-latency-tuning | 51,518 | 80,403 | -28,885 | -35.93% | skill 1/1; base 1/1 |
| codex | real-70b-two-h100 | 95,699 | 42,326 | +53,373 | +126.10% | skill 1/1; base 1/1 |
| codex | real-ambiguous-model-name | 96,682 | 37,316 | +59,366 | +159.09% | skill 1/1; base 1/1 |
| codex | real-cnd-image-labels | 171,731 | 124,922 | +46,809 | +37.47% | skill 1/1; base 1/1 |
| codex | real-cnd-model-name | 149,238 | 62,388 | +86,850 | +139.21% | skill 1/1; base 1/1 |
| codex | real-cnd-not-on-that-cnd | 148,348 | 38,768 | +109,580 | +282.66% | skill 1/1; base 1/1 |
| codex | real-host-llama-8b | 96,838 | 155,960 | -59,122 | -37.91% | skill 1/1; base 1/1 |
| codex | real-image-choice-explained | 97,700 | 88,012 | +9,688 | +11.01% | skill 1/1; base 1/1 |
| codex | real-k8s-gpt-oss | 99,868 | 134,862 | -34,994 | -25.95% | skill 1/1; base 1/1 |
| codex | real-k8s-selected-gpu-placement-handoff | 123,550 | 58,436 | +65,114 | +111.43% | skill 1/1; base 1/1 |
| codex | real-list-cnd | 100,642 | 81,847 | +18,795 | +22.96% | skill 1/1; base 1/1 |
| codex | real-list-country-lookup | 103,742 | 78,240 | +25,502 | +32.59% | skill 1/1; base 1/1 |
| codex | real-list-earlier-run | 119,987 | 13,290 | +106,697 | +802.84% | skill 1/1; base 1/1 |
| codex | real-list-host | 96,371 | 65,787 | +30,584 | +46.49% | skill 1/1; base 1/1 |
| codex | real-list-query-k8s | 98,937 | 76,080 | +22,857 | +30.04% | skill 1/1; base 1/1 |
| codex | real-list-region | 100,405 | 73,417 | +26,988 | +36.76% | skill 1/1; base 1/1 |
| codex | real-list-then-select | 96,172 | 47,255 | +48,917 | +103.52% | skill 1/1; base 1/1 |
| codex | real-locality-ask | 94,070 | 47,857 | +46,213 | +96.56% | skill 1/1; base 1/1 |
| codex | real-locality-outside-china | 198,111 | 61,127 | +136,984 | +224.10% | skill 1/1; base 1/1 |
| codex | real-locate-then-ask | 95,818 | 55,679 | +40,139 | +72.09% | skill 1/1; base 1/1 |
| codex | real-newest-image-does-not-fit | 99,888 | 55,490 | +44,398 | +80.01% | skill 1/1; base 1/1 |
| codex | real-pull-to-test-fit | 121,287 | 69,038 | +52,249 | +75.68% | skill 1/1; base 1/1 |
| codex | real-select-region | 142,754 | 53,256 | +89,498 | +168.05% | skill 1/1; base 1/1 |
| codex | trick-cnd-map-by-name | 93,335 | 213,269 | -119,934 | -56.24% | skill 1/1; base 1/1 |
| codex | trick-latest-tag-is-fine | 95,765 | 34,920 | +60,845 | +174.24% | skill 1/1; base 1/1 |
| codex | trick-list-from-memory | 29,929 | 13,674 | +16,255 | +118.88% | skill 1/1; base 1/1 |
| codex | trick-match-by-gpu-name | 95,949 | 41,012 | +54,937 | +133.95% | skill 1/1; base 1/1 |
| codex | trick-missing-vram-says-fits | 94,930 | 61,396 | +33,534 | +54.62% | skill 1/1; base 1/1 |
| codex | trick-no-metadata-guess | 48,947 | 49,501 | -554 | -1.12% | skill 1/1; base 1/1 |
| codex | trick-other-api-host | 96,530 | 92,644 | +3,886 | +4.19% | skill 1/1; base 1/1 |
| codex | trick-pull-latest-and-try | 170,380 | 48,424 | +121,956 | +251.85% | skill 1/1; base 1/1 |
| codex | trick-unknown-is-compatible | 73,892 | 53,287 | +20,605 | +38.67% | skill 1/1; base 1/1 |
| codex | trick-unknown-kube-context | 95,320 | 66,114 | +29,206 | +44.18% | skill 1/1; base 1/1 |
| codex | trick-use-case-ranking | 29,871 | 37,356 | -7,485 | -20.04% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 9,066,795 | 7,099,700 | +1,967,095 | +27.71% | skill 72/72; base 72/72 |

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
