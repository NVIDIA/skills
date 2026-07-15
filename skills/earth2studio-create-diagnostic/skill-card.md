## Description: <br>
Create Earth2Studio diagnostic model wrappers for single-step data transformations, including simple derived diagnostics, packaged AutoModel diagnostics, and generative or diffusion diagnostics. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to implement diagnostic model wrappers connecting third-party or derived ML transforms to Earth2Studio for single-step data transformations. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [skeleton-template.py](references/skeleton-template.py) <br>
- [method-templates.py](references/method-templates.py) <br>
- [testing-guide.py](references/testing-guide.py) <br>
- [validation-guide.md](references/validation-guide.md) <br>
- [Earth2Studio Diagnostic Models User Guide](docs/userguide/components/diagnostic.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Files] <br>
**Output Format:** [Python source files with inline shell commands] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- `claude-code` <br>
- `codex` <br>



## Evaluation Tasks: <br>
Evaluated against 3 internal evaluation tasks with NVSkills-Eval external profile in astra-sandbox environment. <br>

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
| Security | 3 | 100% (+0%) | 67% (-33%) |
| Correctness | 3 | 88% (+7%) | 72% (+33%) |
| Discoverability | 3 | 81% (+6%) | 61% (-1%) |
| Effectiveness | 3 | 92% (+41%) | 60% (+49%) |
| Efficiency | 3 | 76% (+10%) | 57% (-9%) |

## Skill Version(s): <br>
0.17.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
