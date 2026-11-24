## Description: <br>
Run the full DEFT smart-data-augmentation loop for NVIDIA TAO Grounding DINO object detection: zero-shot baseline inference, KPI analysis, per-class gap analysis, SigLIP embedding of weak images, unique-neighbor mining against a source pool, ODVG dataset staging, and retraining — repeated for a fixed number of iterations. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and ML engineers use this skill to have an agent run an iterative DEFT smart-data-augmentation loop for TAO Grounding DINO object detection (baseline, gap analysis, unique-neighbor mining, ODVG staging, and retraining across N iterations), and to prepare the Co-DETR pseudo-labeled, embedded source pool the loop mines from. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Data layout](references/data-layout.md) <br>
- [Source-pool prep](references/prep-source-pool.md) <br>
- [Pre-Flight checks](references/preflight.md) <br>
- [Pipeline and state](references/pipeline-and-state.md) <br>
- [Scripts and agents](references/scripts-and-agents.md) <br>
- [Grounding DINO overlay](references/grounding-dino.md) <br>
- [Gap analysis overlay](references/tao-analyze-gaps-od-map.md) <br>
- [Image embeddings overlay](references/tao-generate-image-embeddings.md) <br>
- [Mining overlay](references/tao-mine-od-images.md) <br>
- [Stage mined data](references/stage-mined-data.md) <br>
- [Detection KPI overlay](references/tao-analyze-detection-kpi.md) <br>
- [TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Files, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks; run artifacts (deft_state.json, loop_log.jsonl, checkpoints, KPI files, DEFT_Loop_Report.md)] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Verbose train, inference, and Docker output is redirected to files; stage state is committed through commit_stage.py.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (4 positive), 1 attempt per task, each task attempt run in its own isolated sandbox pod (`k8s-sandbox`); dataset digest `sha256:d3fce2a83efb9ec178ecc5fc8a20f8c2280adffa3b279916a073015372c25237` (skill-evaluator-dataset-snapshot/1); evaluator version `1.6.10`; evaluation date 2026-11-24. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from `security` (100%). <br>
- Correctness: Is the answer correct? Scored from `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Scored from `goal_accuracy` (50%) + `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Scored from `skill_efficiency` (50%) + `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Security: unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Skill Execution: whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool Productivity: tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Accuracy: final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Goal Accuracy: whether the user's goal was achieved. <br>
- `behavior_check`: Behavior Check: whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Token Efficiency: actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
**Overall verdict: PASS — Recommended for publication**

### Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 83.4% — baseline ran, but no comparable score was available; uplift unavailable | 63.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 75.0% → 100.0% (+25.0 points) |
| Correctness | 15.0% → 90.0% (+75.0 points) | 20.0% → 70.0% (+50.0 points) |
| Discoverability | 75.0% — baseline ran, but no comparable score was available; uplift unavailable | 23.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 14.4% → 62.5% (+48.1 points) | 16.2% → 27.8% (+11.6 points) |
| Efficiency | 89.7% — baseline ran, but no comparable score was available; uplift unavailable | 96.5% — baseline ran, but no comparable score was available; uplift unavailable |

Baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points. A partial dimension was calculated from only the available configured signals.

#### Comparison details

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.6% → 96.0% (-2.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 93.2% → 93.1% (-0.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 97 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 5 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

Scoring: dimension bands are PASS at 50% or above, NEUTRAL from 40% to below 50%, FAIL below 40%. Overall verdict is PASS only when every configured dimension passes for at least one supported agent; lift is reported as diagnostic evidence and does not override this gate.

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
