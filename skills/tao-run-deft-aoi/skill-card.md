## Description: <br>
Run the full DEFT AOI improvement loop for NVIDIA TAO VisualChangeNet / ChangeNet PCB inspection models: baseline evaluate, RCA, Cosmos AnomalyGen / AMP synthetic defects, k-NN mining, retraining, and deployment gating until FAR / recall KPI targets are met. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and engineers use this skill to run the complete DEFT AOI improvement loop for NVIDIA TAO VisualChangeNet PCB inspection models, iterating through baseline evaluation, root cause analysis, synthetic defect generation, data mining, and retraining until false-accept-rate and recall KPI targets are met. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Pipeline and State](references/pipeline-and-state.md) <br>
- [Pre-Flight Checks](references/preflight.md) <br>
- [Visual ChangeNet](references/visual-changenet.md) <br>
- [Data Layout](references/data-layout.md) <br>
- [PAIDF AnomalyGen](references/paidf-anomalygen.md) <br>
- [Scripts and Agents](references/scripts-and-agents.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, HTML reports, JSON state files] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Produces DEFT_Loop_Report.html, deft_state.json, and loop_log.jsonl as persistent artifacts] <br>

## Evaluation Agents Used: <br>
- Claude Code (`claude-code`) <br>
- Codex (`codex`) <br>



## Evaluation Tasks: <br>
Evaluated against 1 evaluation task in the NVSkills-Eval external profile (astra-sandbox environment). <br>

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
| Correctness | 1 | 100% (+100%) | 88% (+88%) |
| Discoverability | 1 | 83% (+83%) | 62% (+62%) |
| Effectiveness | 1 | 90% (+76%) | 90% (+76%) |
| Efficiency | 1 | 66% (+39%) | 61% (+33%) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
