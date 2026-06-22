## Description: <br>
Run AutoML / hyperparameter optimization (HPO) for NVIDIA TAO networks using AutoMLRunner, handling algorithm selection, WandB experiment tracking, job execution on any TAO SDK platform, result interpretation, and per-recommendation custom evaluation hooks. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and ML engineers use this skill to run automated hyperparameter optimization for NVIDIA TAO models across multiple compute backends (Brev, SLURM, Kubernetes, Docker), selecting search algorithms and interpreting tuning results. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [AutoML Preflight Concepts](references/automl-preflight-concepts.md) <br>
- [AutoML Intent & Algorithms](references/automl-intent-algorithms.md) <br>
- [AutoML Runner Configuration](references/automl-runner-configuration.md) <br>
- [AutoML Advanced Monitoring](references/automl-advanced-monitoring.md) <br>
- [AutoML Examples](references/automl-examples.md) <br>
- [Detailed Guide](references/detailed-guide.md) <br>
- [Skill Info](references/skill_info.yaml) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Python scripts, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash and Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`claude-code`) <br>
- Codex (`codex`) <br>



## Evaluation Tasks: <br>
Evaluated against 1 evaluation task in the NVSkills-Eval external profile on the astra-sandbox environment. <br>

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
| Discoverability | 1 | 100% (+100%) | 72% (+72%) |
| Effectiveness | 1 | 50% (+40%) | 90% (+76%) |
| Efficiency | 1 | 95% (+68%) | 59% (+31%) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
