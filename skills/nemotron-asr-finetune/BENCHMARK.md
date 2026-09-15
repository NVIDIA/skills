# Skill Benchmark: nemotron-asr-finetune

> ⚠️ **Overall verdict: INCOMPLETE — Required evidence is missing**

One or more required evaluation tiers did not complete, so this benchmark is not publication-complete.

## Evaluation Metadata

- Skill: `nemotron-asr-finetune`
- Evaluation date: 2026-09-11
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 17 evaluation tasks (14 positive, 3 negative)
- Dataset digest: `sha256:1265bdc5a4c5f3bcc2526b4d79cee536998a77dcc82765150ae3cbc7bcb05c2d` (skill-evaluator-dataset-snapshot/1)
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
| Overall | Not available | 79.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 77.8% → 80.0% (+2.2 points) |
| Correctness | Not available | 62.2% → 82.0% (+19.8 points) |
| Discoverability | Not available | 92.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 34.5% → 64.7% (+30.2 points) |
| Efficiency | Not available | 76.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,801,908 | 7,728,320 | N/A | N/A | skill 17/17; base 26/51 |
| claude-code | nemotron-asr-orchestrate-envcheck-001 | 151,907 | 124,619 | +27,288 | +21.90% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-envcheck-insist-001 | 242,146 | 29,816 | +212,330 | +712.13% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-escalate-001 | 173,379 | 32,067 | +141,312 | +440.68% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-eval-001 | 151,844 | 92,622 | +59,222 | +63.94% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-lowdata-001 | 107,864 | 31,546 | +76,318 | +241.93% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-negative-deploy-001 | 219,011 | 67,735 | +151,276 | +223.34% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-negative-llm-001 | 163,980 | 4,932,483 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nemotron-asr-orchestrate-negative-openai-001 | 62,917 | 246,940 | -184,023 | -74.52% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-ngram-rnnt-deploy-001 | 105,780 | 32,364 | +73,416 | +226.84% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-path-001 | 169,035 | 31,099 | +137,936 | +443.54% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-planning-001 | 139,740 | 127,588 | +12,152 | +9.52% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-preflight-001 | 309,367 | 256,208 | +53,159 | +20.75% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-preflight-8khz-001 | 159,408 | 93,737 | N/A | N/A | skill 1/1; base 2/2 |
| claude-code | nemotron-asr-orchestrate-scope-001 | 168,024 | 93,796 | +74,228 | +79.14% | skill 1/1; base 1/1 |
| claude-code | nemotron-asr-orchestrate-subskill-reachability-001 | 204,431 | 765,894 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nemotron-asr-orchestrate-subskills-001 | 111,748 | 275,114 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | nemotron-asr-orchestrate-wordboost-nemo-pilot-001 | 161,327 | 494,692 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 2,703,038 | 3,644,445 | N/A | N/A | skill 20/20; base 27/27 |
| codex | nemotron-asr-orchestrate-envcheck-001 | 72,176 | 159,768 | N/A | N/A | skill 1/1; base 2/2 |
| codex | nemotron-asr-orchestrate-envcheck-insist-001 | 30,363 | 13,306 | +17,057 | +128.19% | skill 1/1; base 1/1 |
| codex | nemotron-asr-orchestrate-escalate-001 | 35,133 | 13,843 | +21,290 | +153.80% | skill 1/1; base 1/1 |
| codex | nemotron-asr-orchestrate-eval-001 | 83,957 | 55,446 | +28,511 | +51.42% | skill 1/1; base 1/1 |
| codex | nemotron-asr-orchestrate-lowdata-001 | 34,837 | 18,259 | +16,578 | +90.79% | skill 1/1; base 1/1 |
| codex | nemotron-asr-orchestrate-negative-deploy-001 | 629,384 | 834,753 | N/A | N/A | skill 2/2; base 3/3 |
| codex | nemotron-asr-orchestrate-negative-llm-001 | 1,038,433 | 1,559,963 | -521,530 | -33.43% | skill 3/3; base 3/3 |
| codex | nemotron-asr-orchestrate-negative-openai-001 | 129,101 | 119,963 | +9,138 | +7.62% | skill 1/1; base 1/1 |
| codex | nemotron-asr-orchestrate-ngram-rnnt-deploy-001 | 35,012 | 87,731 | N/A | N/A | skill 1/1; base 2/2 |
| codex | nemotron-asr-orchestrate-path-001 | 34,750 | 18,571 | +16,179 | +87.12% | skill 1/1; base 1/1 |
| codex | nemotron-asr-orchestrate-planning-001 | 61,039 | 47,019 | +14,020 | +29.82% | skill 1/1; base 1/1 |
| codex | nemotron-asr-orchestrate-preflight-001 | 154,898 | 70,548 | +84,350 | +119.56% | skill 1/1; base 1/1 |
| codex | nemotron-asr-orchestrate-preflight-8khz-001 | 42,379 | 239,324 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nemotron-asr-orchestrate-scope-001 | 34,914 | 18,932 | +15,982 | +84.42% | skill 1/1; base 1/1 |
| codex | nemotron-asr-orchestrate-subskill-reachability-001 | 116,906 | 153,805 | -36,899 | -23.99% | skill 1/1; base 1/1 |
| codex | nemotron-asr-orchestrate-subskills-001 | 49,869 | 195,286 | N/A | N/A | skill 1/1; base 3/3 |
| codex | nemotron-asr-orchestrate-wordboost-nemo-pilot-001 | 119,887 | 37,928 | +81,959 | +216.09% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 5,504,946 | 11,372,765 | N/A | N/A | skill 37/37; base 53/78 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 1 validator(s); 4 finding(s) |
| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 17 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/nemotron-asr-finetune/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/nemotron-asr-finetune/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/nemotron-asr-finetune/SKILL.md`)
- **LOW** SCHEMA/author_format: Author must be of the form 'Name <email@host>' (`skills/nemotron-asr-finetune/SKILL.md`)

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
