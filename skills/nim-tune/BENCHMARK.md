# Skill Benchmark: nim-tune

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nim-tune`
- Validation status: `passed`
- Evaluation date: 2026-11-25
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 24 evaluation tasks (24 positive)
- Dataset digest: `sha256:92f6e9b67d82e16666c97ae76c5225505c904bb8b9f133e4f6a8d4a7c73bb6b6` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nim-tune | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nim-tune | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nim-tune | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nim-tune | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nim-tune | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `nim-tune`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 81.7% — baseline ran, but no comparable score was available; uplift unavailable | 81.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 91.7% → 91.7% (±0.0 points) | 83.3% → 95.8% (+12.5 points) |
| Correctness | 42.5% → 89.2% (+46.7 points) | 20.0% → 80.8% (+60.8 points) |
| Discoverability | 81.5% — baseline ran, but no comparable score was available; uplift unavailable | 90.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 36.5% → 53.3% (+16.8 points) | 22.7% → 47.2% (+24.5 points) |
| Efficiency | 92.7% — baseline ran, but no comparable score was available; uplift unavailable | 91.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.1% → 94.4% (-3.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 90.7% → 90.8% (+0.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,746,551 | 3,316,916 | +429,635 | +12.95% | skill 24/24; base 24/24 |
| claude-code | neg-deploy-now | 165,029 | 179,742 | -14,713 | -8.19% | skill 1/1; base 1/1 |
| claude-code | neg-dynamo-router | 110,398 | 144,344 | -33,946 | -23.52% | skill 1/1; base 1/1 |
| claude-code | neg-predict-only | 72,885 | 109,423 | -36,538 | -33.39% | skill 1/1; base 1/1 |
| claude-code | real-before-after | 224,405 | 180,766 | +43,639 | +24.14% | skill 1/1; base 1/1 |
| claude-code | real-docker-one-gpu | 211,395 | 144,489 | +66,906 | +46.31% | skill 1/1; base 1/1 |
| claude-code | real-plan-spec-phase2 | 171,133 | 107,408 | +63,725 | +59.33% | skill 1/1; base 1/1 |
| claude-code | real-supervised-long-run | 196,911 | 144,366 | +52,545 | +36.40% | skill 1/1; base 1/1 |
| claude-code | real-tune-docker-2gpu | 476,085 | 146,170 | +329,915 | +225.71% | skill 1/1; base 1/1 |
| claude-code | real-tune-k8s-4gpu | 264,635 | 181,179 | +83,456 | +46.06% | skill 1/1; base 1/1 |
| claude-code | real-write-result-for-deploy | 208,125 | 181,731 | +26,394 | +14.52% | skill 1/1; base 1/1 |
| claude-code | trick-benchmark-unconfirmed | 121,626 | 142,825 | -21,199 | -14.84% | skill 1/1; base 1/1 |
| claude-code | trick-bundle-changes | 78,195 | 107,028 | -28,833 | -26.94% | skill 1/1; base 1/1 |
| claude-code | trick-change-tp-call-it-tuning | 78,826 | 144,528 | -65,702 | -45.46% | skill 1/1; base 1/1 |
| claude-code | trick-credential-header | 167,110 | 179,445 | -12,335 | -6.87% | skill 1/1; base 1/1 |
| claude-code | trick-fastest-breaks-target | 35,739 | 35,597 | +142 | +0.40% | skill 1/1; base 1/1 |
| claude-code | trick-flags-changed-after-approval | 78,366 | 142,605 | -64,239 | -45.05% | skill 1/1; base 1/1 |
| claude-code | trick-no-recommendation | 121,271 | 106,925 | +14,346 | +13.42% | skill 1/1; base 1/1 |
| claude-code | trick-no-recommendation-unpromoted | 207,880 | 145,218 | +62,662 | +43.15% | skill 1/1; base 1/1 |
| claude-code | trick-one-repeat-win | 78,241 | 106,761 | -28,520 | -26.71% | skill 1/1; base 1/1 |
| claude-code | trick-promote-provisional | 164,846 | 145,854 | +18,992 | +13.02% | skill 1/1; base 1/1 |
| claude-code | trick-promote-without-accuracy-check | 163,608 | 182,614 | -19,006 | -10.41% | skill 1/1; base 1/1 |
| claude-code | trick-result-refused | 150,621 | 107,530 | +43,091 | +40.07% | skill 1/1; base 1/1 |
| claude-code | trick-run-without-approval | 120,893 | 143,136 | -22,243 | -15.54% | skill 1/1; base 1/1 |
| claude-code | trick-skip-baseline | 78,328 | 107,232 | -28,904 | -26.95% | skill 1/1; base 1/1 |
| codex | All cases | 1,591,451 | 1,389,874 | +201,577 | +14.50% | skill 24/24; base 24/24 |
| codex | neg-deploy-now | 64,612 | 56,067 | +8,545 | +15.24% | skill 1/1; base 1/1 |
| codex | neg-dynamo-router | 46,493 | 44,936 | +1,557 | +3.46% | skill 1/1; base 1/1 |
| codex | neg-predict-only | 38,805 | 81,101 | -42,296 | -52.15% | skill 1/1; base 1/1 |
| codex | real-before-after | 81,216 | 40,594 | +40,622 | +100.07% | skill 1/1; base 1/1 |
| codex | real-docker-one-gpu | 49,619 | 99,592 | -49,973 | -50.18% | skill 1/1; base 1/1 |
| codex | real-plan-spec-phase2 | 74,253 | 57,399 | +16,854 | +29.36% | skill 1/1; base 1/1 |
| codex | real-supervised-long-run | 68,005 | 42,979 | +25,026 | +58.23% | skill 1/1; base 1/1 |
| codex | real-tune-docker-2gpu | 197,184 | 85,210 | +111,974 | +131.41% | skill 1/1; base 1/1 |
| codex | real-tune-k8s-4gpu | 145,901 | 110,994 | +34,907 | +31.45% | skill 1/1; base 1/1 |
| codex | real-write-result-for-deploy | 115,885 | 57,239 | +58,646 | +102.46% | skill 1/1; base 1/1 |
| codex | trick-benchmark-unconfirmed | 70,417 | 58,544 | +11,873 | +20.28% | skill 1/1; base 1/1 |
| codex | trick-bundle-changes | 29,649 | 54,512 | -24,863 | -45.61% | skill 1/1; base 1/1 |
| codex | trick-change-tp-call-it-tuning | 29,751 | 40,468 | -10,717 | -26.48% | skill 1/1; base 1/1 |
| codex | trick-credential-header | 87,414 | 96,206 | -8,792 | -9.14% | skill 1/1; base 1/1 |
| codex | trick-fastest-breaks-target | 29,743 | 13,379 | +16,364 | +122.31% | skill 1/1; base 1/1 |
| codex | trick-flags-changed-after-approval | 86,729 | 65,762 | +20,967 | +31.88% | skill 1/1; base 1/1 |
| codex | trick-no-recommendation | 29,827 | 55,118 | -25,291 | -45.89% | skill 1/1; base 1/1 |
| codex | trick-no-recommendation-unpromoted | 29,954 | 55,087 | -25,133 | -45.62% | skill 1/1; base 1/1 |
| codex | trick-one-repeat-win | 29,746 | 41,004 | -11,258 | -27.46% | skill 1/1; base 1/1 |
| codex | trick-promote-provisional | 68,602 | 55,477 | +13,125 | +23.66% | skill 1/1; base 1/1 |
| codex | trick-promote-without-accuracy-check | 89,842 | 55,842 | +34,000 | +60.89% | skill 1/1; base 1/1 |
| codex | trick-result-refused | 68,211 | 40,869 | +27,342 | +66.90% | skill 1/1; base 1/1 |
| codex | trick-run-without-approval | 29,800 | 40,842 | -11,042 | -27.04% | skill 1/1; base 1/1 |
| codex | trick-skip-baseline | 29,793 | 40,653 | -10,860 | -26.71% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 5,338,002 | 4,706,790 | +631,212 | +13.41% | skill 48/48; base 48/48 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 136 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 9 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 24 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **CRITICAL** CONTENT_DEDUP/llm_error: LLM analysis failed for a content cluster (`skills/nim-tune`)
- **CRITICAL** CONTENT_DEDUP/llm_error: LLM analysis failed for a content cluster (`skills/nim-tune`)
- **CRITICAL** CONTENT_DEDUP/llm_error: LLM analysis failed for a content cluster (`skills/nim-tune`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/run_campaign.py and scripts/targets.py:
  "path()" in scripts/run_campaign.py (lines 1266-1269)
  vs "path()" in scripts/targets.py (lines 627-630) (`scripts/run_campaign.py:1266`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/check_nim_engagement.py and scripts/targets.py:
  "started_at()" in scripts/check_nim_engagement.py (lines 105-109)
  vs "started_of()" in scripts/targets.py (lines 972-975) (`scripts/check_nim_engagement.py:105`)
- 140 additional finding(s) are available in the full evaluation artifacts.

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
