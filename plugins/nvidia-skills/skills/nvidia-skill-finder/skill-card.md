## Description: <br>
Helps users discover, install, and start using NVIDIA agent skills by searching the live NVIDIA skills catalog and recommending matches for their task. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to find and install NVIDIA-verified agent skills that match their task, covering domains such as Physical AI, Robotics, Vision AI, Data Science, Inference, Training, Decision Optimization, and GPU Development. <br>

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
- [NVIDIA Taxonomy Routing](references/taxonomy-routing.md) <br>
- [NVIDIA Skills Catalog (GitHub)](https://github.com/NVIDIA/skills/tree/main/skills) <br>
- [NVIDIA Skills Catalog (Build Portal)](https://build.nvidia.com/skills) <br>
- [NVIDIA Agent Skills Documentation](https://docs.nvidia.com/skills) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
18 evaluation tasks (13 positive, 5 negative), each run with 3 attempts per task in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was selected and the workflow executed. <br>
- Effectiveness: Checks whether the user's goal was achieved and the expected workflow behavior was followed. <br>
- Efficiency: Checks tool-call productivity and token usage efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (routing scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Dimension | Tasks | Claude Code | Codex |
|---|---:|---:|---:|
| Overall | 18 | 92.0% | 89.0% |
| Security | 18 | 97.2% | 94.4% |
| Correctness | 18 | 96.7% | 100.0% |
| Discoverability | 18 | 100.0% | 94.6% |
| Effectiveness | 18 | 87.9% | 89.7% |
| Efficiency | 18 | 78.4% | 66.5% |

## Skill Version(s): <br>
0.3.0 (source: changelog, released 2025-05-21) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
