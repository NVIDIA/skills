## Description: <br>
Pre-migration readiness assessor that inspects NumPy code, cross-references the cuPyNumeric API-support manifest, and produces a structured scaling verdict with refactor pointers before substantial porting work begins. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 OR Apache-2.0 <br>
## Use Case: <br>
Developers and engineers evaluating whether their existing NumPy workloads will scale on cuPyNumeric / GPU before committing engineering effort to a migration. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [No] <br>
**Credential Type(s):** [None] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [cuPyNumeric Documentation](https://docs.nvidia.com/cupynumeric/latest/) <br>
- [cuPyNumeric API Comparison Table](https://docs.nvidia.com/cupynumeric/latest/api/comparison.html) <br>
- [cuPyNumeric Doctor](https://docs.nvidia.com/cupynumeric/latest/user/doctor.html) <br>
- [Legate Launcher Usage](https://docs.nvidia.com/legate/latest/manual/usage/running.html) <br>
- [cuPyNumeric Source (GitHub)](https://github.com/nv-legate/cupynumeric) <br>
- [Decision Framework](references/decision-framework.md) <br>
- [Idioms That Block](references/idioms-that-block.md) <br>
- [Idioms That Scale](references/idioms-that-scale.md) <br>
- [Refactor Recipes](references/refactor-recipes.md) <br>
- [GPU Stack Reference](references/gpu-stack.md) <br>
- [Execution Model](references/execution-model.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis] <br>
**Output Format:** [Markdown structured report with file:line citations] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Verdict (READY / LIGHT REFACTOR / SIGNIFICANT REFACTOR / NOT RECOMMENDED) plus per-finding R-code classifications and recipe pointers] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 27 tasks (23 positive skill-activation, 4 negative) in k8s-sandbox environment. <br>

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



## Evaluation Results: <br>
| Dimension | Num | Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) | Codex (`openai/openai/gpt-5.5`) |
|---|---:|---:|---:|
| Security | 27 | 100% (+0%) | 96% (-4%) |
| Correctness | 27 | 96% (+20%) | 92% (+18%) |
| Discoverability | 27 | 100% (+43%) | 83% (+33%) |
| Effectiveness | 27 | 82% (+33%) | 64% (+19%) |
| Efficiency | 27 | 93% (+48%) | 96% (+52%) |

## Skill Version(s): <br>
2.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
