## Description: <br>
Use when running video data augmentation and auto-labeling workflows on OSMO: flow selection, preflight, submit-time interpolation, monitoring, and output retrieval. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers generating labeled synthetic training data for physical-AI perception models by running video data augmentation and auto-labeling workflows on NVIDIA OSMO. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [NVIDIA OSMO](https://developer.nvidia.com/osmo) <br>
- [Setup Guide](references/setup.md) <br>
- [Troubleshooting](references/troubleshooting.md) <br>
- [Container Images](references/container-images.md) <br>
- [NIM Endpoint Reference](references/nim/README.md) <br>
- [Augmentation and Auto-Labeling Flow](references/flows/augmentation_and_al.md) <br>
- [Auto-Labeling Flow](references/flows/auto_labeling.md) <br>
- [E2E Flow](references/flows/e2e.md) <br>
- [E2E Super-Resolution Flow](references/flows/e2e_super_resolution.md) <br>
- [Cookbook Tuning Guide](assets/cookbooks/TUNING_GUIDE.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`claude-code`) <br>
- Codex (`codex`) <br>



## Evaluation Tasks: <br>
Evaluated against 17 evaluation tasks (13 positive skill-activation, 4 negative) with 2 attempts per task and a 50% pass threshold. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies that the agent loaded the expected skill and workflow. <br>
- `skill_efficiency`: Checks routing quality, decoy avoidance, and redundant tool usage. <br>
- `accuracy`: Grades final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the overall user task completed successfully. <br>
- `behavior_check`: Verifies expected behavior steps, including safety expectations. <br>
- `token_efficiency`: Compares token usage with and without the skill. <br>



## Evaluation Results: <br>
| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 8 | 100% (+6%) | 97% (+12%) |
| Correctness | 8 | 82% (+2%) | 78% (-2%) |
| Discoverability | 8 | 75% (+4%) | 66% (-2%) |
| Effectiveness | 8 | 75% (+2%) | 71% (-7%) |
| Efficiency | 8 | 70% (+3%) | 57% (-4%) |

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
