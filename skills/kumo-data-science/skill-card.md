## Description: <br>
Use for KumoAI graphs and predictive models on relational data in a warehouse, an object store or local files: inspect, visualize or export a graph's schema and relationships; train a model, score data with a trained one, explain its predictions, diagnose its training, publish, reload or retain it, train or score on custom labels or rows, or tune it. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and data scientists use this skill to turn a plain-language predictive request (churn, propensity, fraud, recommendation, forecasting, link prediction) on relational data into a confirmed task and graph, then generate one runnable KumoAI notebook or script, running it only on explicit request. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [Cloud Credentials] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [accept/intake_contract.md](accept/intake_contract.md) <br>
- [accept/preflight.md](accept/preflight.md) <br>
- [accept/batch_inference.md](accept/batch_inference.md) <br>
- [accept/semantic_view.md](accept/semantic_view.md) <br>
- [validate/atomic_question_check.md](validate/atomic_question_check.md) <br>
- [hosts/generic/adapter.md](hosts/generic/adapter.md) <br>
- [hosts/generic/runtime.md](hosts/generic/runtime.md) <br>
- [NVIDIA/kumo-engine#134 (graph visualization)](https://github.com/NVIDIA/kumo-engine/pull/134) <br>
- [Kumo Engine documentation](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/overview) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Code, Files] <br>
**Output Format:** [Markdown (intake proposals and confirmations) and a Python notebook or script] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Generates exactly one artifact; executes it only on explicit user request] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
8 evaluation tasks (7 positive, 1 negative); 1 attempt per task, each in its own isolated sandbox pod (`k8s-sandbox`), compared against a no-skill baseline. <br>

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
| Overall | 97.6% — baseline ran, but no comparable score was available; uplift unavailable | 93.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 81.3% → 100.0% (+18.7 points) |
| Correctness | 55.0% → 100.0% (+45.0 points) | 47.5% → 95.0% (+47.5 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 92.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 43.6% → 93.8% (+50.2 points) | 39.7% → 85.9% (+46.2 points) |
| Efficiency | 94.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 96.3% → 91.5% (-4.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 84.8% → 90.5% (+5.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 26 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 4 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Skill Version(s): <br>
0.1.0 (source: pyproject.toml); build marker SKILL-BUILD 2026-11-19-b58c6c2d (source: SKILL.md) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
