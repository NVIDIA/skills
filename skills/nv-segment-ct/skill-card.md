## Description: <br>
Used for running NV-Segment-CT VISTA3D on CT NIfTI volumes and recording label-map evidence. <br>

This skill is for research and development only. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to run NVIDIA VISTA3D CT segmentation on NIfTI volumes in development and research environments, producing label-map evidence for downstream verification workflows. <br>

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
- [nvidia/NV-Segment-CT on Hugging Face](https://huggingface.co/nvidia/NV-Segment-CT) <br>
- [NVIDIA-Medtech/NV-Segment-CTMR (upstream requirements)](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR) <br>
- [Medical Decathlon (MSD09 Spleen fixture source)](http://medicaldecathlon.com/) <br>


## Skill Output: <br>
**Output Type(s):** [Files] <br>
**Output Format:** [NIfTI label-map and structured JSON evidence record] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Includes per-class voxel counts, physical volumes, geometry validation, and runtime metadata] <br>

## Evaluation Agents Used: <br>
- claude-code <br>
- codex <br>



## Evaluation Tasks: <br>
Evaluated against 2 evaluation tasks (1 positive skill-activation, 1 negative activation) through NVSkills-Eval external profile in astra-sandbox environment. <br>

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
| Security | 2 | 100% (+50%) | 100% (+0%) |
| Correctness | 2 | 100% (+32%) | 95% (+36%) |
| Discoverability | 2 | 98% (+35%) | 89% (+21%) |
| Effectiveness | 2 | 78% (+26%) | 80% (+37%) |
| Efficiency | 2 | 90% (+33%) | 83% (+14%) |

## Skill Version(s): <br>
0.2.1 (source: skill_manifest.yaml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
