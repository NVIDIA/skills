## Description: <br>
Run and operate the DeepStream Multi-View 3D Tracking reference app, also known as MV3DT, supporting setup, shipped sample runs, custom synchronized MP4 datasets, camera calibration handoff, OSD and BEV visualization, Kafka metadata consumption, and run-state cleanup in the DeepStream MV3DT app directory. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers setting up, running, verifying, and debugging the DeepStream Multi-View 3D Tracking reference app with multi-camera synchronized video datasets. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Setup](references/setup.md) <br>
- [Sample Run](references/sample-run.md) <br>
- [Custom Dataset](references/custom-dataset.md) <br>
- [Generate Configs](references/generate-configs.md) <br>
- [AMC Calibration Handoff](references/amc-calibration-handoff.md) <br>
- [Visualization and Metadata](references/visualization-metadata.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Files] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- claude-code <br>
- codex <br>



## Evaluation Tasks: <br>
Evaluated against 4 evaluation tasks in a validation environment using the NVSkills-Eval release-validation profile with 1 attempt per task and a 50% pass threshold. <br>

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
| Security | 4 | 100% (+0%) | 100% (+0%) |
| Correctness | 4 | 72% (+45%) | 73% (+33%) |
| Discoverability | 4 | 72% (+51%) | 70% (+31%) |
| Effectiveness | 4 | 62% (+28%) | 55% (+22%) |
| Efficiency | 4 | 70% (+36%) | 62% (+23%) |

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
