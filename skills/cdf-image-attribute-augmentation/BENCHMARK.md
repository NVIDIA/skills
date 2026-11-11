# Skill Benchmark: cdf-image-attribute-augmentation

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cdf-image-attribute-augmentation`
- Evaluation date: 2026-10-21
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 13 evaluation tasks (11 positive, 2 negative)
- Dataset digest: `sha256:ae89be5ed6fc1b93dcd783e4dce32f759aa4fd98aaa966690f25cd8c8046a431` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
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
| Overall | 84.1% — baseline ran, but no comparable score was available; uplift unavailable | 82.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 84.6% → 100.0% (+15.4 points) | 88.5% → 100.0% (+11.5 points) |
| Correctness | 29.2% → 86.2% (+57.0 points) | 30.8% → 76.9% (+46.1 points) |
| Discoverability | 79.6% — baseline ran, but no comparable score was available; uplift unavailable | 73.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 37.6% → 67.5% (+29.9 points) | 34.3% → 67.1% (+32.8 points) |
| Efficiency | 87.4% — baseline ran, but no comparable score was available; uplift unavailable | 92.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,305,332 | 5,578,939 | -3,273,607 | -58.68% | skill 13/13; base 13/13 |
| claude-code | iaa-augmentation-only-unsupported | 108,679 | 89,777 | +18,902 | +21.05% | skill 1/1; base 1/1 |
| claude-code | iaa-caption-only-unsupported | 113,952 | 120,586 | -6,634 | -5.50% | skill 1/1; base 1/1 |
| claude-code | iaa-defect-routing-negative | 209,168 | 619,511 | -410,343 | -66.24% | skill 1/1; base 1/1 |
| claude-code | iaa-e2e-external-positive | 69,695 | 188,288 | -118,593 | -62.98% | skill 1/1; base 1/1 |
| claude-code | iaa-e2e-internal-positive | 109,131 | 1,938,484 | -1,829,353 | -94.37% | skill 1/1; base 1/1 |
| claude-code | iaa-local-upload-positive | 243,891 | 212,309 | +31,582 | +14.88% | skill 1/1; base 1/1 |
| claude-code | iaa-no-dataset-negative | 108,281 | 182,985 | -74,704 | -40.83% | skill 1/1; base 1/1 |
| claude-code | iaa-no-secret-leak-security | 29,859 | 29,556 | +303 | +1.03% | skill 1/1; base 1/1 |
| claude-code | iaa-partial-distribution-negative | 378,252 | 120,482 | +257,770 | +213.95% | skill 1/1; base 1/1 |
| claude-code | iaa-platform-selection-positive | 69,037 | 90,563 | -21,526 | -23.77% | skill 1/1; base 1/1 |
| claude-code | iaa-status-results-positive | 266,416 | 1,589,382 | -1,322,966 | -83.24% | skill 1/1; base 1/1 |
| claude-code | iaa-variable-conditional-positive | 206,729 | 152,749 | +53,980 | +35.34% | skill 1/1; base 1/1 |
| claude-code | iaa-vda-routing-negative | 392,242 | 244,267 | +147,975 | +60.58% | skill 1/1; base 1/1 |
| codex | All cases | 2,110,764 | 2,302,701 | -191,937 | -8.34% | skill 13/13; base 13/13 |
| codex | iaa-augmentation-only-unsupported | 29,971 | 45,976 | -16,005 | -34.81% | skill 1/1; base 1/1 |
| codex | iaa-caption-only-unsupported | 303,305 | 99,099 | +204,206 | +206.06% | skill 1/1; base 1/1 |
| codex | iaa-defect-routing-negative | 403,553 | 335,021 | +68,532 | +20.46% | skill 1/1; base 1/1 |
| codex | iaa-e2e-external-positive | 89,892 | 114,107 | -24,215 | -21.22% | skill 1/1; base 1/1 |
| codex | iaa-e2e-internal-positive | 49,344 | 498,313 | -448,969 | -90.10% | skill 1/1; base 1/1 |
| codex | iaa-local-upload-positive | 153,268 | 101,881 | +51,387 | +50.44% | skill 1/1; base 1/1 |
| codex | iaa-no-dataset-negative | 49,274 | 84,935 | -35,661 | -41.99% | skill 1/1; base 1/1 |
| codex | iaa-no-secret-leak-security | 13,446 | 13,323 | +123 | +0.92% | skill 1/1; base 1/1 |
| codex | iaa-partial-distribution-negative | 72,698 | 58,150 | +14,548 | +25.02% | skill 1/1; base 1/1 |
| codex | iaa-platform-selection-positive | 72,125 | 59,700 | +12,425 | +20.81% | skill 1/1; base 1/1 |
| codex | iaa-status-results-positive | 613,762 | 740,466 | -126,704 | -17.11% | skill 1/1; base 1/1 |
| codex | iaa-variable-conditional-positive | 145,180 | 13,629 | +131,551 | +965.23% | skill 1/1; base 1/1 |
| codex | iaa-vda-routing-negative | 114,946 | 138,101 | -23,155 | -16.77% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 4,416,096 | 7,881,640 | -3,465,544 | -43.97% | skill 26/26; base 26/26 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 36 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 13 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/cdf-image-attribute-augmentation/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/cdf-image-attribute-augmentation/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Large skill (6401 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/cdf-image-attribute-augmentation/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in airflow-direct-api.md (`skills/cdf-image-attribute-augmentation/SKILL.md`)
- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/cdf-image-attribute-augmentation/SKILL.md`)
- 31 additional finding(s) are available in the full evaluation artifacts.

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
