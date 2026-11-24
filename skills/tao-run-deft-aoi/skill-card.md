## Description: <br>
Run the full DEFT AOI improvement loop for NVIDIA TAO VisualChangeNet / ChangeNet PCB inspection models: baseline evaluate, RCA, Cosmos AnomalyGen / AMP synthetic defects, k-NN mining, retraining, and deployment gating against a customer-defined primary metric and optional constraints. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and ML engineers use this skill to have an agent autonomously run the iterative DEFT AOI improvement loop (baseline evaluation, root-cause analysis, synthetic defect generation, data mining, retraining, and deployment gating) for NVIDIA TAO VisualChangeNet / ChangeNet PCB automated-optical-inspection models until a customer-defined KPI target is met or the iteration budget is exhausted, including air-gapped runs with pre-staged assets. <br>

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
- [air-gap.md](references/air-gap.md) <br>
- [data-layout.md](references/data-layout.md) <br>
- [metric-contract.md](references/metric-contract.md) <br>
- [network-bootstrap.md](references/network-bootstrap.md) <br>
- [pipeline-and-state.md](references/pipeline-and-state.md) <br>
- [preflight.md](references/preflight.md) <br>
- [prepare-for-inference.md](references/prepare-for-inference.md) <br>
- [scripts-and-agents.md](references/scripts-and-agents.md) <br>
- [tao-analyze-gaps-visual-changenet.md](references/tao-analyze-gaps-visual-changenet.md) <br>
- [tao-generate-anomalies.md](references/tao-generate-anomalies.md) <br>
- [tao-mine-aoi-images.md](references/tao-mine-aoi-images.md) <br>
- [tao-route-visual-changenet-samples.md](references/tao-route-visual-changenet-samples.md) <br>
- [visual-changenet.md](references/visual-changenet.md) <br>
- [REPORT_RENDERING.md](references/REPORT_RENDERING.md) <br>
- [NVIDIA TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Files, Analysis] <br>
**Output Format:** [Markdown status updates with inline bash commands; deft_state.json run state; HTML loop report (DEFT_Loop_Report.html)] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [State transitions are committed atomically through bundled scripts; verbose container and training logs are redirected to files under the results directory.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
3 evaluation tasks (3 positive): tao-run-deft-aoi-basic, tao-run-deft-aoi-customer-metric, and tao-run-deft-aoi-resume-state; 1 attempt per task, each in an isolated k8s-sandbox pod, compared against a no-skill baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from `security` (100%). <br>
- Correctness: Is the answer correct? Scored from `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Scored from `goal_accuracy` (50%) + `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Scored from `skill_efficiency` (50%) + `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
> ✅ **Overall verdict: PASS — Recommended for publication**

### Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 87.1% — baseline ran, but no comparable score was available; uplift unavailable | 80.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 66.7% → 100.0% (+33.3 points) |
| Correctness | 40.0% → 100.0% (+60.0 points) | 60.0% → 93.3% (+33.3 points) |
| Discoverability | 91.7% — baseline ran, but no comparable score was available; uplift unavailable | 55.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 17.8% → 52.2% (+34.4 points) | 37.2% → 61.1% (+23.9 points) |
| Efficiency | 91.7% — baseline ran, but no comparable score was available; uplift unavailable | 94.9% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 98.0% → 92.9% (-5.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 89.2% → 89.9% (+0.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 104 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 5 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 3 task(s) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
