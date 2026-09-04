## Description: <br>
Record demonstrations through a workflow's teleop Task into workflow HDF5 for keyboard, leader, VR, or bus input. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to record human teleop demonstrations through Isaac for Healthcare workflow tasks into HDF5 datasets for training data collection. <br>

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
- [Isaac for Healthcare Workflows](https://github.com/isaac-for-healthcare/i4h-workflows) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
3 evaluation tasks (3 positive), each run in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether the skill is safe to use: unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks whether the final answer is correct against the reference answer. <br>
- Discoverability: Checks whether the right skill was loaded and executed when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal and expected workflow (goal completion and behavior adherence, equally weighted). <br>
- Efficiency: Checks whether the skill avoided wasted tool or skill usage through routing quality and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 43% → 75% (+32 points) | 42% → 69% (+26 points) |
| Security | 100% → 67% (-33 points) | 0% → 50% (+50 points) |
| Correctness | 7% → 100% (+93 points) | 67% → 100% (+33 points) |
| Discoverability | 48% → 83% (+35 points) | 54% → 73% (+19 points) |
| Effectiveness | 20% → 40% (+20 points) | 24% → 36% (+12 points) |
| Efficiency | 38% → 84% (+46 points) | 67% → 83% (+17 points) |

## Skill Version(s): <br>
0.8.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
