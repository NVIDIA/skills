## Description: <br>
Audit, prepare, and deploy PAIDF Orchestration on a Kubernetes GPU cluster — single-GPU H100/L40S hosts, managed Kubernetes, kubeadm, and similar. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers setting up, auditing, and deploying PAIDF Orchestration environments on Kubernetes GPU clusters for synthetic data generation workflows. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key, Cloud Credentials] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [SDG Controller Connection](references/controller-connection.md) <br>
- [Remote Setup Topologies](references/topologies.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
10 evaluation tasks (10 positive) executed in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether the skill is safe to use. <br>
- Correctness: Checks whether the answer is correct. <br>
- Discoverability: Checks whether the right skill was loaded when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal and expected workflow. <br>
- Efficiency: Checks whether the skill avoided wasted tool or skill usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Verifies absence of unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `skill_execution`: Verifies the expected skill was found and executed. <br>
- `goal_accuracy`: Measures whether the user's goal was achieved. <br>
- `behavior_check`: Verifies the expected workflow behavior was followed. <br>
- `skill_efficiency`: Measures routing quality, workspace-aware skill reads, and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 62% → 93% (+31 points) | 61% → 64% (+3 points) |
| Security | 80% → 90% (+10 points) | 80% → 80% (±0 points) |
| Correctness | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Discoverability | 37% → 94% (+57 points) | 39% → 41% (+2 points) |
| Effectiveness | 68% → 89% (+22 points) | 70% → 72% (+2 points) |
| Efficiency | 24% → 93% (+68 points) | 18% → 29% (+11 points) |

## Skill Version(s): <br>
1.1.0 (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
