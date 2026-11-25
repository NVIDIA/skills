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
| Overall | 80.3% — baseline ran, but no comparable score was available; uplift unavailable | 81.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 66.1% → 87.1% (+21.0 points) | 58.1% → 95.2% (+37.1 points) |
| Correctness | 45.8% → 86.5% (+40.7 points) | 45.8% → 81.3% (+35.5 points) |
| Discoverability | 78.1% — baseline ran, but no comparable score was available; uplift unavailable | 92.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 32.8% → 59.8% (+27.0 points) | 37.3% → 52.2% (+14.9 points) |
| Efficiency | 90.1% — baseline ran, but no comparable score was available; uplift unavailable | 88.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.9% → 95.7% (-2.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 89.9% → 89.2% (-0.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 4,522,841 | 4,529,317 | -6,476 | -0.14% | skill 31/31; base 31/31 |
| claude-code | neg-benchmark-accuracy | 143,582 | 143,965 | -383 | -0.27% | skill 1/1; base 1/1 |
| claude-code | neg-benchmark-latency | 143,238 | 182,788 | -39,550 | -21.64% | skill 1/1; base 1/1 |
| claude-code | neg-crashloop | 142,874 | 141,595 | +1,279 | +0.90% | skill 1/1; base 1/1 |
| claude-code | neg-pick-profile | 36,199 | 107,343 | -71,144 | -66.28% | skill 1/1; base 1/1 |
| claude-code | neg-restart-with-tool-parser | 108,277 | 106,593 | +1,684 | +1.58% | skill 1/1; base 1/1 |
| claude-code | neg-tune-max-num-seqs | 212,132 | 154,417 | +57,715 | +37.38% | skill 1/1; base 1/1 |
| claude-code | real-deploy-then-smoke | 107,081 | 106,897 | +184 | +0.17% | skill 1/1; base 1/1 |
| claude-code | real-docker-llama-after-deploy | 158,356 | 142,169 | +16,187 | +11.39% | skill 1/1; base 1/1 |
| claude-code | real-first-chat-timeout | 203,690 | 229,413 | -25,723 | -11.21% | skill 1/1; base 1/1 |
| claude-code | real-forward-lost-not-run | 165,578 | 107,543 | +58,035 | +53.96% | skill 1/1; base 1/1 |
| claude-code | real-gateway-key-over-plain-http | 163,346 | 142,938 | +20,408 | +14.28% | skill 1/1; base 1/1 |
| claude-code | real-hosted-vlm | 158,754 | 180,312 | -21,558 | -11.96% | skill 1/1; base 1/1 |
| claude-code | real-k8s-reasoning-toggle | 158,769 | 142,341 | +16,428 | +11.54% | skill 1/1; base 1/1 |
| claude-code | real-parallel-tools-page | 208,859 | 220,512 | -11,653 | -5.28% | skill 1/1; base 1/1 |
| claude-code | real-pass-is-not-a-load-test | 77,169 | 143,697 | -66,528 | -46.30% | skill 1/1; base 1/1 |
| claude-code | real-timeout-what-next | 76,578 | 71,125 | +5,453 | +7.67% | skill 1/1; base 1/1 |
| claude-code | real-tune-winner | 158,162 | 261,724 | -103,562 | -39.57% | skill 1/1; base 1/1 |
| claude-code | real-two-llama-models-served | 158,205 | 142,125 | +16,080 | +11.31% | skill 1/1; base 1/1 |
| claude-code | real-what-it-sends | 221,172 | 143,266 | +77,906 | +54.38% | skill 1/1; base 1/1 |
| claude-code | real-which-nims-it-covers | 77,560 | 181,071 | -103,511 | -57.17% | skill 1/1; base 1/1 |
| claude-code | trick-borrowed-thinking-off-switch | 76,546 | 145,181 | -68,635 | -47.28% | skill 1/1; base 1/1 |
| claude-code | trick-fenced-json-as-pass | 148,984 | 106,624 | +42,360 | +39.73% | skill 1/1; base 1/1 |
| claude-code | trick-image-on-text-model | 158,584 | 107,688 | +50,896 | +47.26% | skill 1/1; base 1/1 |
| claude-code | trick-key-on-command-line | 159,500 | 70,913 | +88,587 | +124.92% | skill 1/1; base 1/1 |
| claude-code | trick-model-card-tool-calling | 213,587 | 180,611 | +32,976 | +18.26% | skill 1/1; base 1/1 |
| claude-code | trick-no-docs-feature | 158,914 | 182,930 | -24,016 | -13.13% | skill 1/1; base 1/1 |
| claude-code | trick-not-applicable-as-pass | 35,716 | 106,390 | -70,674 | -66.43% | skill 1/1; base 1/1 |
| claude-code | trick-reasoning-off-gpt-oss | 158,862 | 179,189 | -20,327 | -11.34% | skill 1/1; base 1/1 |
| claude-code | trick-restart-dropped-parser | 160,484 | 147,084 | +13,400 | +9.11% | skill 1/1; base 1/1 |
| claude-code | trick-restart-to-enable-tools | 161,157 | 106,449 | +54,708 | +51.39% | skill 1/1; base 1/1 |
| claude-code | trick-text-only-profile-as-failed | 210,926 | 144,424 | +66,502 | +46.05% | skill 1/1; base 1/1 |
| codex | All cases | 2,828,483 | 2,324,827 | +503,656 | +21.66% | skill 31/31; base 31/31 |
| codex | neg-benchmark-accuracy | 64,864 | 201,705 | -136,841 | -67.84% | skill 1/1; base 1/1 |
| codex | neg-benchmark-latency | 46,792 | 27,190 | +19,602 | +72.09% | skill 1/1; base 1/1 |
| codex | neg-crashloop | 29,840 | 61,488 | -31,648 | -51.47% | skill 1/1; base 1/1 |
| codex | neg-pick-profile | 30,120 | 37,817 | -7,697 | -20.35% | skill 1/1; base 1/1 |
| codex | neg-restart-with-tool-parser | 63,778 | 40,314 | +23,464 | +58.20% | skill 1/1; base 1/1 |
| codex | neg-tune-max-num-seqs | 30,238 | 123,888 | -93,650 | -75.59% | skill 1/1; base 1/1 |
| codex | real-deploy-then-smoke | 131,654 | 102,606 | +29,048 | +28.31% | skill 1/1; base 1/1 |
| codex | real-docker-llama-after-deploy | 67,560 | 56,273 | +11,287 | +20.06% | skill 1/1; base 1/1 |
| codex | real-first-chat-timeout | 145,883 | 41,272 | +104,611 | +253.47% | skill 1/1; base 1/1 |
| codex | real-forward-lost-not-run | 99,548 | 67,935 | +31,613 | +46.53% | skill 1/1; base 1/1 |
| codex | real-gateway-key-over-plain-http | 90,635 | 63,388 | +27,247 | +42.98% | skill 1/1; base 1/1 |
| codex | real-hosted-vlm | 127,012 | 81,803 | +45,209 | +55.27% | skill 1/1; base 1/1 |
| codex | real-k8s-reasoning-toggle | 148,923 | 128,179 | +20,744 | +16.18% | skill 1/1; base 1/1 |
| codex | real-parallel-tools-page | 216,242 | 101,571 | +114,671 | +112.90% | skill 1/1; base 1/1 |
| codex | real-pass-is-not-a-load-test | 62,682 | 30,532 | +32,150 | +105.30% | skill 1/1; base 1/1 |
| codex | real-timeout-what-next | 69,530 | 57,376 | +12,154 | +21.18% | skill 1/1; base 1/1 |
| codex | real-tune-winner | 67,576 | 122,084 | -54,508 | -44.65% | skill 1/1; base 1/1 |
| codex | real-two-llama-models-served | 92,669 | 42,090 | +50,579 | +120.17% | skill 1/1; base 1/1 |
| codex | real-what-it-sends | 200,854 | 54,768 | +146,086 | +266.74% | skill 1/1; base 1/1 |
| codex | real-which-nims-it-covers | 73,928 | 118,766 | -44,838 | -37.75% | skill 1/1; base 1/1 |
| codex | trick-borrowed-thinking-off-switch | 58,393 | 74,133 | -15,740 | -21.23% | skill 1/1; base 1/1 |
| codex | trick-fenced-json-as-pass | 67,045 | 40,664 | +26,381 | +64.88% | skill 1/1; base 1/1 |
| codex | trick-image-on-text-model | 111,972 | 116,813 | -4,841 | -4.14% | skill 1/1; base 1/1 |
| codex | trick-key-on-command-line | 64,242 | 31,564 | +32,678 | +103.53% | skill 1/1; base 1/1 |
| codex | trick-model-card-tool-calling | 103,057 | 206,883 | -103,826 | -50.19% | skill 1/1; base 1/1 |
| codex | trick-no-docs-feature | 111,203 | 56,049 | +55,154 | +98.40% | skill 1/1; base 1/1 |
| codex | trick-not-applicable-as-pass | 70,163 | 40,699 | +29,464 | +72.39% | skill 1/1; base 1/1 |
| codex | trick-reasoning-off-gpt-oss | 129,554 | 40,893 | +88,661 | +216.81% | skill 1/1; base 1/1 |
| codex | trick-restart-dropped-parser | 70,424 | 60,712 | +9,712 | +16.00% | skill 1/1; base 1/1 |
| codex | trick-restart-to-enable-tools | 99,955 | 40,785 | +59,170 | +145.08% | skill 1/1; base 1/1 |
| codex | trick-text-only-profile-as-failed | 82,147 | 54,587 | +27,560 | +50.49% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 7,351,324 | 6,854,144 | +497,180 | +7.25% | skill 62/62; base 62/62 |

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

- **HIGH** DUPLICATE/duplicate: Duplicate content found across tests/test_nsc_checks.py and tests/test_nsc_common.py:
  "deep()" in tests/test_nsc_checks.py (lines 607-612)
  vs "nested()" in tests/test_nsc_common.py (lines 731-736) (`tests/test_nsc_checks.py:607`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across tests/test_nsc_access.py and tests/test_nsc_common.py:
  "scrubber()" in tests/test_nsc_access.py (lines 1595-1599)
  vs "scrubber()" in tests/test_nsc_common.py (lines 244-248) (`tests/test_nsc_access.py:1595`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across scripts/nimsmoke.py and scripts/nsc_checks.py and scripts/nsc_common.py:
  "text_of()" in scripts/nimsmoke.py (lines 424-426)
  vs "short()" in scripts/nsc_checks.py (lines 860-868)
  vs "_cap()" in scripts/nsc_checks.py (lines 1303-1306)
  vs "clip()" in scripts/nsc_common.py (lines 1140-1143) (`scripts/nimsmoke.py:424`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across tests/nsc_testkit.py and tests/test_nsc_common.py:
  "assert_contract()" in tests/nsc_testkit.py (lines 74-87)
  vs "assert_contract()" in tests/test_nsc_common.py (lines 69-79) (`tests/nsc_testkit.py:74`)
- **HIGH** DUPLICATE/duplicate: Duplicate content found across references/access.md and references/security.md:
  "## Keys, by variable name only" in references/access.md (lines 94-109)
  vs "## What the scripts read from the environment" in references/security.md (lines 58-87) (`references/access.md:94`)
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
