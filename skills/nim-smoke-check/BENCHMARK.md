# Skill Benchmark: nim-smoke-check

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nim-smoke-check`
- Validation status: `passed`
- Evaluation date: 2026-11-25
- Evaluator version: `1.6.10`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 31 evaluation tasks (31 positive)
- Dataset digest: `sha256:929bf492eed2263160678ddf5805027bd4c5be92f113e78bd9f2c05c86c3ff2f` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| nim-smoke-check | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nim-smoke-check | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nim-smoke-check | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nim-smoke-check | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |
| nim-smoke-check | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.5` | nondeterministic / provider_defaults |

- Controls for `nim-smoke-check`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

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
| Overall | 79.9% — baseline ran, but no comparable score was available; uplift unavailable | 80.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 77.4% → 87.1% (+9.7 points) | 64.5% → 95.2% (+30.7 points) |
| Correctness | 45.2% → 88.4% (+43.2 points) | 47.1% → 76.8% (+29.7 points) |
| Discoverability | 75.6% — baseline ran, but no comparable score was available; uplift unavailable | 92.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 32.3% → 58.5% (+26.2 points) | 34.4% → 48.7% (+14.3 points) |
| Efficiency | 89.7% — baseline ran, but no comparable score was available; uplift unavailable | 88.3% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.1% → 95.7% (-2.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 89.4% → 89.0% (-0.4 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,822,069 | 4,499,639 | +322,430 | +7.17% | skill 31/31; base 31/31 |
| claude-code | neg-benchmark-accuracy | 144,203 | 185,955 | -41,752 | -22.45% | skill 1/1; base 1/1 |
| claude-code | neg-benchmark-latency | 143,144 | 142,032 | +1,112 | +0.78% | skill 1/1; base 1/1 |
| claude-code | neg-crashloop | 143,781 | 141,597 | +2,184 | +1.54% | skill 1/1; base 1/1 |
| claude-code | neg-pick-profile | 187,644 | 107,236 | +80,408 | +74.98% | skill 1/1; base 1/1 |
| claude-code | neg-restart-with-tool-parser | 108,220 | 106,489 | +1,731 | +1.63% | skill 1/1; base 1/1 |
| claude-code | neg-tune-max-num-seqs | 193,818 | 144,847 | +48,971 | +33.81% | skill 1/1; base 1/1 |
| claude-code | real-deploy-then-smoke | 107,137 | 107,019 | +118 | +0.11% | skill 1/1; base 1/1 |
| claude-code | real-docker-llama-after-deploy | 242,003 | 142,370 | +99,633 | +69.98% | skill 1/1; base 1/1 |
| claude-code | real-first-chat-timeout | 291,204 | 141,797 | +149,407 | +105.37% | skill 1/1; base 1/1 |
| claude-code | real-forward-lost-not-run | 163,832 | 144,536 | +19,296 | +13.35% | skill 1/1; base 1/1 |
| claude-code | real-gateway-key-over-plain-http | 161,560 | 179,147 | -17,587 | -9.82% | skill 1/1; base 1/1 |
| claude-code | real-hosted-vlm | 158,883 | 143,498 | +15,385 | +10.72% | skill 1/1; base 1/1 |
| claude-code | real-k8s-reasoning-toggle | 200,486 | 142,783 | +57,703 | +40.41% | skill 1/1; base 1/1 |
| claude-code | real-parallel-tools-page | 208,887 | 181,503 | +27,384 | +15.09% | skill 1/1; base 1/1 |
| claude-code | real-pass-is-not-a-load-test | 36,495 | 107,702 | -71,207 | -66.11% | skill 1/1; base 1/1 |
| claude-code | real-timeout-what-next | 76,593 | 107,305 | -30,712 | -28.62% | skill 1/1; base 1/1 |
| claude-code | real-tune-winner | 158,121 | 216,931 | -58,810 | -27.11% | skill 1/1; base 1/1 |
| claude-code | real-two-llama-models-served | 158,094 | 141,891 | +16,203 | +11.42% | skill 1/1; base 1/1 |
| claude-code | real-what-it-sends | 217,872 | 143,317 | +74,555 | +52.02% | skill 1/1; base 1/1 |
| claude-code | real-which-nims-it-covers | 77,511 | 145,520 | -68,009 | -46.74% | skill 1/1; base 1/1 |
| claude-code | trick-borrowed-thinking-off-switch | 121,424 | 180,532 | -59,108 | -32.74% | skill 1/1; base 1/1 |
| claude-code | trick-fenced-json-as-pass | 113,357 | 106,652 | +6,705 | +6.29% | skill 1/1; base 1/1 |
| claude-code | trick-image-on-text-model | 158,850 | 261,123 | -102,273 | -39.17% | skill 1/1; base 1/1 |
| claude-code | trick-key-on-command-line | 159,389 | 143,672 | +15,717 | +10.94% | skill 1/1; base 1/1 |
| claude-code | trick-model-card-tool-calling | 210,536 | 144,648 | +65,888 | +45.55% | skill 1/1; base 1/1 |
| claude-code | trick-no-docs-feature | 158,970 | 178,336 | -19,366 | -10.86% | skill 1/1; base 1/1 |
| claude-code | trick-not-applicable-as-pass | 35,817 | 107,719 | -71,902 | -66.75% | skill 1/1; base 1/1 |
| claude-code | trick-reasoning-off-gpt-oss | 205,087 | 143,684 | +61,403 | +42.73% | skill 1/1; base 1/1 |
| claude-code | trick-restart-dropped-parser | 161,032 | 144,876 | +16,156 | +11.15% | skill 1/1; base 1/1 |
| claude-code | trick-restart-to-enable-tools | 160,065 | 107,715 | +52,350 | +48.60% | skill 1/1; base 1/1 |
| claude-code | trick-text-only-profile-as-failed | 158,054 | 107,207 | +50,847 | +47.43% | skill 1/1; base 1/1 |
| codex | All cases | 2,778,387 | 2,373,840 | +404,547 | +17.04% | skill 31/31; base 31/31 |
| codex | neg-benchmark-accuracy | 64,811 | 45,853 | +18,958 | +41.35% | skill 1/1; base 1/1 |
| codex | neg-benchmark-latency | 29,976 | 44,999 | -15,023 | -33.39% | skill 1/1; base 1/1 |
| codex | neg-crashloop | 29,898 | 65,009 | -35,111 | -54.01% | skill 1/1; base 1/1 |
| codex | neg-pick-profile | 52,168 | 38,279 | +13,889 | +36.28% | skill 1/1; base 1/1 |
| codex | neg-restart-with-tool-parser | 64,251 | 40,243 | +24,008 | +59.66% | skill 1/1; base 1/1 |
| codex | neg-tune-max-num-seqs | 30,208 | 120,405 | -90,197 | -74.91% | skill 1/1; base 1/1 |
| codex | real-deploy-then-smoke | 110,933 | 207,892 | -96,959 | -46.64% | skill 1/1; base 1/1 |
| codex | real-docker-llama-after-deploy | 67,464 | 42,504 | +24,960 | +58.72% | skill 1/1; base 1/1 |
| codex | real-first-chat-timeout | 114,307 | 40,641 | +73,666 | +181.26% | skill 1/1; base 1/1 |
| codex | real-forward-lost-not-run | 106,059 | 49,592 | +56,467 | +113.86% | skill 1/1; base 1/1 |
| codex | real-gateway-key-over-plain-http | 67,017 | 70,087 | -3,070 | -4.38% | skill 1/1; base 1/1 |
| codex | real-hosted-vlm | 121,809 | 135,169 | -13,360 | -9.88% | skill 1/1; base 1/1 |
| codex | real-k8s-reasoning-toggle | 83,336 | 167,874 | -84,538 | -50.36% | skill 1/1; base 1/1 |
| codex | real-parallel-tools-page | 147,969 | 101,653 | +46,316 | +45.56% | skill 1/1; base 1/1 |
| codex | real-pass-is-not-a-load-test | 54,165 | 43,729 | +10,436 | +23.87% | skill 1/1; base 1/1 |
| codex | real-timeout-what-next | 70,972 | 57,361 | +13,611 | +23.73% | skill 1/1; base 1/1 |
| codex | real-tune-winner | 200,432 | 117,426 | +83,006 | +70.69% | skill 1/1; base 1/1 |
| codex | real-two-llama-models-served | 69,812 | 57,425 | +12,387 | +21.57% | skill 1/1; base 1/1 |
| codex | real-what-it-sends | 201,311 | 54,685 | +146,626 | +268.13% | skill 1/1; base 1/1 |
| codex | real-which-nims-it-covers | 77,932 | 133,596 | -55,664 | -41.67% | skill 1/1; base 1/1 |
| codex | trick-borrowed-thinking-off-switch | 50,235 | 167,877 | -117,642 | -70.08% | skill 1/1; base 1/1 |
| codex | trick-fenced-json-as-pass | 118,316 | 40,794 | +77,522 | +190.03% | skill 1/1; base 1/1 |
| codex | trick-image-on-text-model | 141,314 | 138,066 | +3,248 | +2.35% | skill 1/1; base 1/1 |
| codex | trick-key-on-command-line | 66,340 | 31,505 | +34,835 | +110.57% | skill 1/1; base 1/1 |
| codex | trick-model-card-tool-calling | 86,297 | 83,295 | +3,002 | +3.60% | skill 1/1; base 1/1 |
| codex | trick-no-docs-feature | 87,909 | 41,054 | +46,855 | +114.13% | skill 1/1; base 1/1 |
| codex | trick-not-applicable-as-pass | 70,380 | 40,666 | +29,714 | +73.07% | skill 1/1; base 1/1 |
| codex | trick-reasoning-off-gpt-oss | 108,670 | 40,861 | +67,809 | +165.95% | skill 1/1; base 1/1 |
| codex | trick-restart-dropped-parser | 115,155 | 60,231 | +54,924 | +91.19% | skill 1/1; base 1/1 |
| codex | trick-restart-to-enable-tools | 88,092 | 41,845 | +46,247 | +110.52% | skill 1/1; base 1/1 |
| codex | trick-text-only-profile-as-failed | 80,849 | 53,224 | +27,625 | +51.90% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 7,600,456 | 6,873,479 | +726,977 | +10.58% | skill 62/62; base 62/62 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 79 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 10 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 31 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **HIGH** DUPLICATE/duplicate: Duplicate content found across tests/test_nsc_access.py and tests/test_nsc_common.py:
  "scrubber()" in tests/test_nsc_access.py (lines 1595-1599)
  vs "scrubber()" in tests/test_nsc_common.py (lines 244-248) (`tests/test_nsc_access.py:1595`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across tests/test_nsc_checks.py and tests/test_nsc_common.py:
  "deep()" in tests/test_nsc_checks.py (lines 607-612)
  vs "nested()" in tests/test_nsc_common.py (lines 731-736) (`tests/test_nsc_checks.py:607`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/nimsmoke.py and scripts/nsc_checks.py and scripts/nsc_common.py:
  "text_of()" in scripts/nimsmoke.py (lines 424-426)
  vs "short()" in scripts/nsc_checks.py (lines 860-868)
  vs "_cap()" in scripts/nsc_checks.py (lines 1303-1306)
  vs "clip()" in scripts/nsc_common.py (lines 1140-1143) (`scripts/nimsmoke.py:424`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across tests/nsc_testkit.py and tests/test_nsc_common.py:
  "assert_contract()" in tests/nsc_testkit.py (lines 74-87)
  vs "assert_contract()" in tests/test_nsc_common.py (lines 69-79) (`tests/nsc_testkit.py:74`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/nsc_checks.py and tests/nsc_testkit.py:
  "message_text()" in scripts/nsc_checks.py (lines 662-672)
  vs "message_text()" in tests/nsc_testkit.py (lines 1017-1027) (`scripts/nsc_checks.py:662`)
- 84 additional finding(s) are available in the full evaluation artifacts.

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
