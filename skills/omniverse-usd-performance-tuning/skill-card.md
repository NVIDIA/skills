## Description: <br>
Top-level workflow skill for USD performance diagnosis and optimization that handles slow loading, high memory, low FPS, and broad scene-optimization requests. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers working with NVIDIA Omniverse USD scenes who need to diagnose and resolve performance issues such as slow loading, high memory usage, low FPS, or GPU crashes, and optimize scenes through structured profiling, validation, and operation workflows. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Workflow Reference](references/workflow.md) <br>
- [Skill Map](references/skill-map.md) <br>
- [USD Structure Assessment](references/usd-structure-assessment/README.md) <br>
- [USD Optimize Run Operations](references/usd-optimize-run-operations/README.md) <br>
- [USD Validation Runner](references/usd-validation-runner/README.md) <br>
- [Optimization Report](references/optimization-report/README.md) <br>
- [Setup USD Performance Tuning](references/setup-usd-performance-tuning/README.md) <br>
- [Operations Reference](references/operations/README.md) <br>
- [Profile Stage](references/profile-stage/README.md) <br>
- [Compare Profiles](references/compare-profiles/README.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions, Files] <br>
**Output Format:** [Structured JSON report, Markdown summary, and HTML report] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Produces optimized USD stage, JSON optimization report conforming to optimization-report schema, Markdown summary, and rendered HTML report] <br>

## Evaluation Agents Used: <br>
- Claude Code (`claude-code`) <br>
- Codex (`codex`) <br>



## Evaluation Tasks: <br>
Evaluated against 9 evaluation tasks (8 positive skill-activation, 1 negative) via NVSkills-Eval external profile in astra-sandbox environment. <br>

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
| Security | 8 | 100% (+0%) | 100% (+0%) |
| Correctness | 8 | 62% (+14%) | 81% (+27%) |
| Discoverability | 8 | 74% (+22%) | 85% (+28%) |
| Effectiveness | 8 | 34% (+8%) | 63% (+29%) |
| Efficiency | 8 | 64% (+15%) | 74% (+24%) |

## Testing Completed: <br>
**[x] Agent Red-Teaming** <br>
**[ ] Network Security** <br>
**[ ] Product Security** <br>

## Skill Version(s): <br>
0.1.0 (source: frontmatter, pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
