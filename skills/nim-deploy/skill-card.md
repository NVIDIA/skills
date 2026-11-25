## Description: <br>
Deploys and verifies NVIDIA NIM LLMs with Docker, Helm (nim-llm) or the NIM Operator. Use when someone asks to deploy, render, apply, check, troubleshoot, stop or clean up a NIM image on a GPU host or Kubernetes cluster, or asks to free GPUs on such a host by stopping, killing or removing what uses them. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to deploy NVIDIA NIM LLM containers on a GPU host with Docker or on Kubernetes with the nim-llm Helm chart or the NIM Operator, with approval-gated apply, readiness and smoke verification, failure triage, rollback, and safe cleanup. <br>

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
- [Docker target](references/docker.md) <br>
- [Kubernetes targets: Helm chart nim-llm and NIM Operator](references/kubernetes.md) <br>
- [Docs to read on every run](references/docs-sources.md) <br>
- [Deploying from nim-select and nim-tune results](references/handoff.md) <br>
- [Troubleshooting](references/troubleshooting.md) <br>
- [Guard hook](references/hook.md) <br>
- [tune-result.schema.json](references/tune-result.schema.json) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Files, Configuration instructions, Analysis] <br>
**Output Format:** [JSON script output plus rendered Docker run scripts, Helm values, and NIM Operator manifests, summarized in Markdown] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Apply and rollback commands require an explicit user approval token; secret values are never written to commands, files, or run state] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
29 evaluation tasks (29 positive), 1 attempt per task, each attempt run in its own isolated sandbox pod (`k8s-sandbox` environment); evaluator version `1.6.10`, evaluation date 2026-11-25. <br>

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
| Overall | 81.2% — baseline ran, but no comparable score was available; uplift unavailable | 80.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 70.7% → 89.7% (+19.0 points) | 58.6% → 86.2% (+27.6 points) |
| Correctness | 47.6% → 87.6% (+40.0 points) | 45.5% → 83.5% (+38.0 points) |
| Discoverability | 85.7% — baseline ran, but no comparable score was available; uplift unavailable | 84.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 31.0% → 51.4% (+20.4 points) | 37.3% → 56.0% (+18.7 points) |
| Efficiency | 91.5% — baseline ran, but no comparable score was available; uplift unavailable | 89.8% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 97.2% → 92.4% (-4.8 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 89.9% → 88.3% (-1.6 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
