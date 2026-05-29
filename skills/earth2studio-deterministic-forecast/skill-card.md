## Description: <br>
Build deterministic forecast scripts with Earth2Studio (model, data source, IO, inference). <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers building deterministic (single-member) weather forecast inference scripts using Earth2Studio, covering model selection, data source compatibility, IO backend choice, and complete script generation. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Earth2Studio Prognostic Models](https://nvidia.github.io/earth2studio/modules/models_px.html) <br>
- [Earth2Studio Data Sources (Analysis)](https://nvidia.github.io/earth2studio/modules/datasources_analysis.html) <br>
- [Earth2Studio Data Sources (Forecast)](https://nvidia.github.io/earth2studio/modules/datasources_forecast.html) <br>
- [Earth2Studio IO Backends](https://nvidia.github.io/earth2studio/modules/io.html) <br>
- [earth2studio.run.deterministic source](https://github.com/NVIDIA/earth2studio/blob/main/earth2studio/run.py) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands] <br>
**Output Format:** [Markdown with inline Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`claude-code`) <br>
- Codex (`codex`) <br>



## Evaluation Tasks: <br>
Evaluated against 3 internal evaluation tasks (positive skill-activation cases, 2 attempts per task, 50% pass threshold). <br>

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
| Security | 6 | 100% (+0%) | 100% (+42%) |
| Correctness | 6 | 88% (+0%) | 81% (+2%) |
| Discoverability | 6 | 68% (-2%) | 55% (+5%) |
| Effectiveness | 6 | 91% (-1%) | 81% (-2%) |
| Efficiency | 6 | 57% (+0%) | 43% (+1%) |

## Skill Version(s): <br>
0.16.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
