## Description: <br>
Kubernetes execution platform — submits TAO container jobs as single-pod k8s Jobs with NVIDIA GPU scheduling. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to submit TAO container training and inference jobs on Kubernetes clusters (EKS, GKE, AKS, or on-prem) with NVIDIA GPU scheduling, including single-node and multi-node distributed training. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [Cloud Credentials, API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [skill_info.yaml](references/skill_info.yaml) <br>
- [NVIDIA GPU Operator Getting Started](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html) <br>
- [Kubernetes Indexed Job Completion Mode](https://kubernetes.io/docs/concepts/workloads/controllers/job/#completion-mode) <br>
- [PyTorch Distributed Elastic Run](https://pytorch.org/docs/stable/elastic/run.html) <br>
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Code] <br>
**Output Format:** [Markdown with inline bash and Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
1 evaluation task (1 positive), each attempt ran in its own isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Detects unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 37% → 95% (+58 points) | 34% → 95% (+61 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 0% → 100% (+100 points) | 40% → 100% (+60 points) |
| Discoverability | 50% → 100% (+50 points) | 0% → 94% (+94 points) |
| Effectiveness | 0% → 100% (+100 points) | 32% → 83% (+52 points) |
| Efficiency | 33% → 75% (+42 points) | 0% → 100% (+100 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
