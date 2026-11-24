## Description: <br>
Use for KumoAI graphs and predictive models on relational data in a warehouse, an object store or local files: inspect, visualize or export a graph's schema and relationships; train a model, score data with a trained one, explain its predictions, diagnose its training, publish, reload or retain it, train or score on custom labels or rows, or tune it. Covers churn, propensity, fraud, recommendation, forecasting and link prediction. For a new model it checks that the request names the data, outcome and time window, confirms the inferred tables and task, then generates one notebook or script; it runs it only when asked, in the notebook or as a host job. Also use it, before any general jobs or workspace skill, for a job, schedule, run or cluster that runs a Kumo notebook or scores a model published from one: creating, scheduling, pausing, submitting or debugging it. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers, data scientists, and ML engineers use this skill to turn plain-language predictive requests on relational data (churn, propensity, fraud, recommendation, forecasting, link prediction) into a confirmed task and graph, then generate one runnable KumoAI notebook or script; it also routes graph inspection, explanations, diagnostics, persistence, custom populations, tuning, and Kumo job requests. <br>

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
- [accept/semantic_view.md](accept/semantic_view.md) <br>
- [accept/batch_inference.md](accept/batch_inference.md) <br>
- [validate/atomic_question_check.md](validate/atomic_question_check.md) <br>
- [hosts/generic/adapter.md](hosts/generic/adapter.md) <br>
- [hosts/generic/runtime.md](hosts/generic/runtime.md) <br>
- [Kumo Engine documentation](https://docs.nvidia.com/sdgm/kumo-engine/get-started-with-kumo-engine/overview) <br>
- [kumo-engine on PyPI](https://pypi.org/project/kumo-engine/) <br>
- [NVIDIA/kumo-engine#134 (graph visualization)](https://github.com/NVIDIA/kumo-engine/pull/134) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Code, Files] <br>
**Output Format:** [Markdown intake proposals and confirmations; one Jupyter notebook (train then predict) or Python script (batch inference)] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Code is generated only after graph and task confirmation; a training notebook contains exactly one fit call; the artifact runs only on explicit user request.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
8 evaluation tasks (7 positive, 1 negative); 1 attempt per task, each in its own isolated sandbox pod (`k8s-sandbox`); evaluator version `1.6.10`, evaluated 2026-11-20. <br>

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
| Overall | 99.2% — baseline ran, but no comparable score was available; uplift unavailable | 95.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 93.8% → 100.0% (+6.2 points) |
| Correctness | 50.0% → 100.0% (+50.0 points) | 45.0% → 97.5% (+52.5 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 90.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 42.4% → 100.0% (+57.6 points) | 36.9% → 93.8% (+56.9 points) |
| Efficiency | 95.9% — baseline ran, but no comparable score was available; uplift unavailable | 95.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

#### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 96.8% → 91.7% (-5.1 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 83.2% → 90.4% (+7.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 26 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 4 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Skill Version(s): <br>
0.1.0 (source: pyproject.toml; git SHA a8190ac6, committed 2026-11-20) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
