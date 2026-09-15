## Description: <br>
Used for generating synthetic T1, T2, FLAIR, SWI, or MRA brain MRI volumes with NV-Generate-CTMR MR-Brain v1. Not for production training data. <br>

This skill is for research and development only. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers generating synthetic brain MRI volumes for research, development, and engineering evaluation using the NV-Generate-CTMR rflow-mr-brain v1 workflow. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [FOV and Downloads](references/fov-and-downloads.md) <br>
- [NVIDIA-Medtech/NV-Generate-CTMR (upstream)](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR/tree/da438fec6484cdb6f421f8c7051d954ebefff730) <br>


## Skill Output: <br>
**Output Type(s):** [Files, Analysis] <br>
**Output Format:** [NIfTI volumes and JSON result summary] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
3 evaluation tasks (3 positive), each with 3 attempts per task in isolated k8s-sandbox pods. Dataset digest: sha256:cd7314ca. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use — checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was selected, decoys were avoided, and the workflow executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal (goal_accuracy 50% + behavior_check 50%). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage (tool productivity 50% + token efficiency 50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code | Codex |
|---|---:|---:|
| Overall | 97.3% | 97.6% |
| Security | 100.0% | 100.0% |
| Correctness | 20.0% → 100.0% (+80.0 pts) | 36.0% → 100.0% (+64.0 pts) |
| Discoverability | 91.7% | 93.3% |
| Effectiveness | 18.4% → 96.7% (+78.3 pts) | 38.0% → 96.7% (+58.7 pts) |
| Efficiency | 98.0% | 98.2% |

## Skill Version(s): <br>
0.1.0 (source: skill_manifest.yaml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
