## Description: <br>
Compare BioIR vs OSS folding performance: model.forward() latency, GPU use, lDDT and DockQ with matched inputs; use for benchmark setup, execution or audits, excluding affinity heads and Ray throughput. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers working on BioIR (BioNeMo Inference Runtime) use this skill to benchmark BioIR against the pinned open-source implementation of a structure-folding model (Boltz-1/2, OpenFold2 / AlphaFold2, OpenFold3, Protenix-v2). It sets up isolated environments, locks matched inputs and config, times serial one-sample model.forward() calls with GPU telemetry, scores lDDT and DockQ, and produces a comparison report. <br>

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
- [workflow.md](workflow.md) <br>
- [measurement.md](measurement.md) <br>
- [environment.md](environment.md) <br>
- [samples.md](samples.md) <br>
- [msa.md](msa.md) <br>
- [templates.md](templates.md) <br>
- [no-pipeline.md](no-pipeline.md) <br>
- [deepspeed-evoformer.md](deepspeed-evoformer.md) <br>
- [bundle.md](bundle.md) <br>
- [Boltz-2 profile](models/boltz2.md) <br>
- [OpenFold2 / AlphaFold2 profile](models/of2.md) <br>
- [OpenFold3 profile](models/of3.md) <br>
- [Protenix-v2 profile](models/protenix.md) <br>
- [BioIR support matrix](../../docs/ref/support-matrix.md) <br>
- [BioIR API reference](../../docs/ref/api.md) <br>
- [aqlaboratory/openfold v2.2.0](https://github.com/aqlaboratory/openfold/tree/v2.2.0) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Files, Analysis] <br>
**Output Format:** [Python benchmark harnesses, JSON result and config artifacts, latency plots, and a Markdown comparison report] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Every latency, GPU power, clock-rate, or speedup figure cites an executed command, exit code, and artifact path] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
11 evaluation tasks (9 positive, 2 negative), 1 attempt per task, each attempt in its own isolated sandbox pod (`k8s-sandbox`); dataset digest `sha256:84f4a60929fe1591ebfdb376f2313a0123288921678895e37f156e774ec52c2b` (skill-evaluator-dataset-snapshot/1); evaluator version `1.6.10`, evaluated 2026-11-18. Baseline is the same task attempted without the target skill. <br>

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
> ✅ **Overall verdict: PASS — Recommended for publication**

### Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | Not available | 91.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 86.4% → 100.0% (+13.6 points) |
| Correctness | Not available | 85.5% → 94.6% (+9.1 points) |
| Discoverability | Not available | 87.2% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 83.2% → 82.8% (-0.4 points) |
| Efficiency | Not available | 91.5% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 86.5% → 88.2% (+1.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 44 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 6 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 11 task(s) |

## Skill Version(s): <br>
0.2.1 (source: repo CHANGELOG; git SHA 5fe1677b, committed 2026-11-18) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
