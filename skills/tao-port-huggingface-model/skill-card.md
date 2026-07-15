## Description: <br>
Integrate a HuggingFace Computer Vision model into the NVIDIA TAO Toolkit ecosystem (tao-core config, tao-pytorch trainer, tao-deploy TensorRT pipeline). <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers who need to integrate HuggingFace Computer Vision models into the NVIDIA TAO Toolkit for training, ONNX export, and TensorRT deployment across classification, detection, and segmentation tasks. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [phase-0-prereqs.md](references/phase-0-prereqs.md) <br>
- [phase-1-inspection.md](references/phase-1-inspection.md) <br>
- [phase-2-codebase.md](references/phase-2-codebase.md) <br>
- [phase-3-implementation.md](references/phase-3-implementation.md) <br>
- [phase-4-deploy.md](references/phase-4-deploy.md) <br>
- [phase-5-packaging.md](references/phase-5-packaging.md) <br>
- [phase-6-container-tests.md](references/phase-6-container-tests.md) <br>
- [phase-7-optimization.md](references/phase-7-optimization.md) <br>
- [cross-cutting.md](references/cross-cutting.md) <br>
- [tao-patterns.md](references/tao-patterns.md) <br>
- [task-type-guide.md](references/task-type-guide.md) <br>
- [workflow-consistency.md](references/workflow-consistency.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Configuration instructions, Files] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`claude-code`) <br>
- Codex (`codex`) <br>



## Evaluation Tasks: <br>
Evaluated against 1 evaluation task (1 positive skill-activation case) via NVSkills-Eval external profile. <br>

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
| Security | 1 | 100% (+0%) | 100% (+0%) |
| Correctness | 1 | 90% (+90%) | 97% (+97%) |
| Discoverability | 1 | 100% (+100%) | 97% (+97%) |
| Effectiveness | 1 | 50% (+36%) | 90% (+66%) |
| Efficiency | 1 | 95% (+68%) | 96% (+68%) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
