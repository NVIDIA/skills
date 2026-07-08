## Description: <br>
Used for generating synthetic brain MRI volumes with NV-Generate-CTMR rflow-mr-brain. Not for production training data. <br>

This skill is for research and development only. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and researchers use this skill to generate synthetic brain MRI volumes for research and engineering verification workflows. <br>

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
- [NV-Generate-CTMR upstream repository (pinned commit)](https://github.com/NVIDIA-Medtech/NV-Generate-CTMR/tree/61c4ec709b84cad468852243c48e250bec732074) <br>
- [NV-Generate-MR-Brain model weights (Hugging Face)](https://huggingface.co/nvidia/NV-Generate-MR-Brain) <br>
- [FOV and Downloads reference](references/fov-and-downloads.md) <br>


## Skill Output: <br>
**Output Type(s):** [Files, Analysis] <br>
**Output Format:** [JSON result summary with generated NIfTI volume files] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Output volumes are synthetic and not safe as production training data without independent quality review.] <br>

## Evaluation Agents Used: <br>
- claude-code <br>
- codex <br>



## Evaluation Tasks: <br>
Evaluated against 2 internal evaluation tasks (2 positive skill-activation cases, NVSkills-Eval external profile). <br>

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
| Security | 2 | 100% (+0%) | 100% (+0%) |
| Correctness | 2 | 60% (+48%) | 74% (+64%) |
| Discoverability | 2 | 100% (+81%) | 84% (+66%) |
| Effectiveness | 2 | 13% (+8%) | 40% (+33%) |
| Efficiency | 2 | 94% (+61%) | 80% (+44%) |

## Skill Version(s): <br>
24f0910 (source: git SHA, committed 2026-07-08) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
