# Skill Benchmark: nemotron-speech

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `nemotron-speech`
- Evaluation date: 2026-09-15
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 18 evaluation tasks (14 positive, 4 negative)
- Dataset digest: `sha256:7da18a129d0ad5efdc392d764333668f68f9acf996152684bc2464427afdb20f` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 3
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

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
| Overall | 89.9% — baseline ran, but no comparable score was available; uplift unavailable | 82.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 72.5% → 88.9% (+16.4 points) | 70.8% → 86.1% (+15.3 points) |
| Correctness | 82.0% → 94.4% (+12.4 points) | 81.7% → 93.3% (+11.6 points) |
| Discoverability | 98.6% — baseline ran, but no comparable score was available; uplift unavailable | 91.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 59.0% → 89.2% (+30.2 points) | 54.2% → 81.5% (+27.3 points) |
| Efficiency | 78.5% — baseline ran, but no comparable score was available; uplift unavailable | 58.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,496,556 | 5,790,207 | N/A | N/A | skill 18/18; base 20/20 |
| claude-code | nemotron-speech-asr-cloud-001 | 275,733 | 523,312 | -247,579 | -47.31% | skill 1/1; base 1/1 |
| claude-code | nemotron-speech-asr-self-hosted-001 | 225,009 | 1,162,589 | -937,580 | -80.65% | skill 1/1; base 1/1 |
| claude-code | nemotron-speech-custom-asr-001 | 149,303 | 132,568 | +16,735 | +12.62% | skill 1/1; base 1/1 |
| claude-code | nemotron-speech-model-selection-001 | 205,338 | 31,182 | +174,156 | +558.51% | skill 1/1; base 1/1 |
| claude-code | nemotron-speech-negative-generic-docker-001 | 133,807 | 133,258 | +549 | +0.41% | skill 1/1; base 1/1 |
| claude-code | nemotron-speech-negative-ipa-linguistics-001 | 29,451 | 29,309 | +142 | +0.48% | skill 1/1; base 1/1 |
| claude-code | nemotron-speech-negative-openai-whisper-001 | 123,117 | 89,260 | +33,857 | +37.93% | skill 1/1; base 1/1 |
| claude-code | nemotron-speech-negative-outlook-001 | 29,512 | 29,688 | -176 | -0.59% | skill 1/1; base 1/1 |
| claude-code | nemotron-speech-nmt-001 | 193,699 | 164,816 | +28,883 | +17.52% | skill 1/1; base 1/1 |
| claude-code | nemotron-speech-pipelines-001 | 155,687 | 717,418 | -561,731 | -78.30% | skill 1/1; base 1/1 |
| claude-code | nemotron-speech-readiness-001 | 190,228 | 94,200 | +96,028 | +101.94% | skill 1/1; base 1/1 |
| claude-code | nemotron-speech-setup-001 | 142,280 | 536,116 | -393,836 | -73.46% | skill 1/1; base 1/1 |
| claude-code | nemotron-speech-tts-001 | 831,670 | 760,538 | +71,132 | +9.35% | skill 1/1; base 1/1 |
| claude-code | nemotron-speech-tts-custom-001 | 151,309 | 31,688 | +119,621 | +377.50% | skill 1/1; base 1/1 |
| claude-code | nemotron-speech-tts-pipelines-001 | 157,524 | 144,911 | +12,613 | +8.70% | skill 1/1; base 1/1 |
| claude-code | nemotron-speech-tts-pipelines-zeroshot-guardrail-001 | 109,560 | 127,755 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nemotron-speech-tts-pronunciation-001 | 288,754 | 1,048,043 | -759,289 | -72.45% | skill 1/1; base 1/1 |
| claude-code | nemotron-speech-tts-pronunciation-002 | 104,575 | 33,556 | +71,019 | +211.64% | skill 1/1; base 1/1 |
| codex | All cases | 2,316,460 | 6,658,586 | N/A | N/A | skill 18/18; base 24/24 |
| codex | nemotron-speech-asr-cloud-001 | 85,171 | 46,504 | +38,667 | +83.15% | skill 1/1; base 1/1 |
| codex | nemotron-speech-asr-self-hosted-001 | 143,099 | 57,026 | +86,073 | +150.94% | skill 1/1; base 1/1 |
| codex | nemotron-speech-custom-asr-001 | 143,112 | 343,743 | -200,631 | -58.37% | skill 1/1; base 1/1 |
| codex | nemotron-speech-model-selection-001 | 81,120 | 33,502 | +47,618 | +142.13% | skill 1/1; base 1/1 |
| codex | nemotron-speech-negative-generic-docker-001 | 223,358 | 268,462 | -45,104 | -16.80% | skill 1/1; base 1/1 |
| codex | nemotron-speech-negative-ipa-linguistics-001 | 13,579 | 13,361 | +218 | +1.63% | skill 1/1; base 1/1 |
| codex | nemotron-speech-negative-openai-whisper-001 | 70,207 | 86,919 | -16,712 | -19.23% | skill 1/1; base 1/1 |
| codex | nemotron-speech-negative-outlook-001 | 13,507 | 13,449 | +58 | +0.43% | skill 1/1; base 1/1 |
| codex | nemotron-speech-nmt-001 | 65,803 | 17,838 | +47,965 | +268.89% | skill 1/1; base 1/1 |
| codex | nemotron-speech-pipelines-001 | 109,947 | 351,790 | -241,843 | -68.75% | skill 1/1; base 1/1 |
| codex | nemotron-speech-readiness-001 | 227,228 | 143,369 | +83,859 | +58.49% | skill 1/1; base 1/1 |
| codex | nemotron-speech-setup-001 | 83,565 | 35,905 | +47,660 | +132.74% | skill 1/1; base 1/1 |
| codex | nemotron-speech-tts-001 | 270,432 | 3,370,697 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nemotron-speech-tts-custom-001 | 84,585 | 72,367 | +12,218 | +16.88% | skill 1/1; base 1/1 |
| codex | nemotron-speech-tts-pipelines-001 | 84,395 | 57,548 | +26,847 | +46.65% | skill 1/1; base 1/1 |
| codex | nemotron-speech-tts-pipelines-zeroshot-guardrail-001 | 112,172 | 103,556 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nemotron-speech-tts-pronunciation-001 | 416,487 | 1,568,708 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nemotron-speech-tts-pronunciation-002 | 88,693 | 73,842 | +14,851 | +20.11% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 5,813,016 | 12,448,793 | N/A | N/A | skill 36/36; base 44/44 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 79 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 18 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **CRITICAL** CONTENT_DEDUP/llm_cluster_member_limit: A Tier 2 cluster exceeds the LLM member limit. (`skills/nemotron-speech`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/nemotron-speech/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/nemotron-speech/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in asr-custom.md (`skills/nemotron-speech/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/nemotron-speech/SKILL.md`)
- 75 additional finding(s) are available in the full evaluation artifacts.

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
