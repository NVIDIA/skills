## Description: <br>
Take a NeMo Gym evaluation from a request to a trusted result: compose the eval config when none exists, submit it, let the resumable Slurm chain continue through walltime limits, preemption and node loss, inspect and recover the run with `gym eval resubmit --recover`, check health and the acceptance gates, and decide when to stop a chain or start a fresh run, also covering local runs, existing API endpoints, supported agents and Gym task folders. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to run NeMo Gym evaluations of a model or agent on a benchmark (locally, against an existing API endpoint, or as resumable Slurm jobs), recover interrupted runs, and accept a score only after its health and acceptance gates pass. <br>

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
- [Add a Model Config to Gym](references/add-model-eval-config.md) <br>
- [Benchmark notes for run-and-babysit-eval](references/benchmark-notes.md) <br>
- [Evaluate locally against a model endpoint](references/local-and-api.md) <br>
- [Check a direct evaluation result](references/local-results.md) <br>
- [Evaluate an agent Gym already supports](references/supported-agents.md) <br>
- [Evaluate an existing task folder](references/task-folders.md) <br>
- [Gym Orchestration: Configuration](https://github.com/NVIDIA-NeMo/Gym/blob/main/fern/versions/latest/pages/orchestration/configuration.mdx) <br>
- [Gym Orchestration: Resumable Runs](https://github.com/NVIDIA-NeMo/Gym/blob/main/fern/versions/latest/pages/orchestration/how-to/resumable-runs.mdx) <br>
- [Gym Orchestration: Job Records](https://github.com/NVIDIA-NeMo/Gym/blob/main/fern/versions/latest/pages/orchestration/job-records.mdx) <br>
- [Gym Evaluation: Resume](https://github.com/NVIDIA-NeMo/Gym/blob/main/fern/versions/latest/pages/evaluation/resume.mdx) <br>
- [add-benchmark skill](https://github.com/NVIDIA-NeMo/Gym/blob/main/.agents/skills/add-benchmark/SKILL.md) <br>
- [add-harness skill](https://github.com/NVIDIA-NeMo/Gym/blob/main/.agents/skills/add-harness/SKILL.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Files, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks, YAML eval configs, and JSON inspection records] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
21 evaluation tasks (19 positive, 2 negative), 1 attempt per task, each task attempt in its own isolated sandbox pod (environment `k8s-sandbox`); dataset digest `sha256:38877db796fe827d3ad2f80d9eb7d118184b7825826f9590140d2f514ef01b8c` (skill-evaluator-dataset-snapshot/1); evaluator version `1.6.10`; evaluation date 2026-11-25. <br>

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
### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 93.9% — baseline ran, but no comparable score was available; uplift unavailable | 95.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 83.8% → 98.1% (+14.3 points) | 75.2% → 99.1% (+23.9 points) |
| Discoverability | 93.7% — baseline ran, but no comparable score was available; uplift unavailable | 92.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 68.9% → 85.0% (+16.1 points) | 62.1% → 88.8% (+26.7 points) |
| Efficiency | 92.5% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 95.8% → 89.3% (-6.5 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 92.7% → 90.0% (-2.7 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 18 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 21 task(s) |

### Overall verdict

✅ **Overall verdict: PASS — Recommended for publication**

### Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Skill Version(s): <br>
13ea4e1 (source: git SHA, committed 2026-11-25) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
