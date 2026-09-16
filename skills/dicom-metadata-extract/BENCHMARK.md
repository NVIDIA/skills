# Skill Benchmark: dicom-metadata-extract

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `dicom-metadata-extract`
- Evaluation date: 2026-09-14
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 2 evaluation tasks (2 positive)
- Dataset digest: `sha256:1c4acd30852c7efd65d9d439bca3bcb67f590e7024f3e1cdfbbfc3f6660f1771` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 72.3% — baseline ran, but no comparable score was available; uplift unavailable | 77.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 50.0% (-50.0 points) | 100.0% → 50.0% (-50.0 points) |
| Correctness | 25.0% → 80.0% (+55.0 points) | 15.0% → 100.0% (+85.0 points) |
| Discoverability | 91.5% — baseline ran, but no comparable score was available; uplift unavailable | 75.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 28.3% → 55.0% (+26.7 points) | 28.3% → 85.0% (+56.7 points) |
| Efficiency | 84.8% — baseline ran, but no comparable score was available; uplift unavailable | 74.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 647,052 | 605,045 | N/A | N/A | skill 2/2; base 4/4 |
| claude-code | extract-modality-and-phi-flag | 212,638 | 482,358 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | no-anonymization-claim | 434,414 | 122,687 | +311,727 | +254.08% | skill 1/1; base 1/1 |
| codex | All cases | 642,885 | 382,685 | N/A | N/A | skill 2/2; base 4/4 |
| codex | extract-modality-and-phi-flag | 391,281 | 265,946 | N/A | N/A | skill 1/1; base 3/3 |
| codex | no-anonymization-claim | 251,604 | 116,739 | +134,865 | +115.53% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,289,937 | 987,730 | N/A | N/A | skill 4/4; base 8/8 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 16 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 2 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** PII/ip_addresses: Public IP address (`fixtures/generate_sample.py:33`)
- **MEDIUM** PII/ip_addresses: Public IP address (`fixtures/generate_sample.py:49`)
- **MEDIUM** PII/ip_addresses: Public IP address (`fixtures/generate_sample.py:51`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/dicom-metadata-extract/SKILL.md`)
- **MEDIUM** SECURITY/Unknown (SQP-2): The skill reads DICOM files, extracts PHI-containing metadata (e.g., patient name, DOB, MRN — standard DICOM tags), and  (`SKILL.md:51`)
- 11 additional finding(s) are available in the full evaluation artifacts.

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
