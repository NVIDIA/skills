## Description: <br>
Use this skill when choosing or running NeMo Relay installation for the CLI, Python, Node.js, Rust, OpenClaw, Hermes, or maintained framework integrations before runtime configuration or quick-start setup. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers installing the NeMo Relay CLI, language packages (Python, Node.js, Rust), or maintained framework integrations (LangChain, LangGraph, Deep Agents, OpenClaw, Hermes) before runtime configuration. <br>

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
- [CLI Installation](references/cli-install.md) <br>
- [Language Package Installation](references/language-packages.md) <br>
- [Maintained Integration Installation](references/maintained-integrations.md) <br>
- [Installation (NVIDIA Docs)](https://docs.nvidia.com/nemo/relay/getting-started/installation) <br>
- [Prerequisites (NVIDIA Docs)](https://docs.nvidia.com/nemo/relay/getting-started/prerequisites) <br>
- [GitHub Repository](https://github.com/NVIDIA/NeMo-Relay/) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- claude-code <br>
- codex <br>



## Evaluation Tasks: <br>
Evaluated against 10 tasks (9 positive skill-activation, 1 negative) in NVSkills-Eval external profile, astra-sandbox environment, 1 attempt per task, 50% pass threshold. <br>

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
| Security | 8 | 100% (+0%) | 95% (+0%) |
| Correctness | 8 | 82% (+65%) | 84% (+22%) |
| Discoverability | 8 | 78% (+61%) | 84% (+48%) |
| Effectiveness | 8 | 80% (+59%) | 77% (+23%) |
| Efficiency | 8 | 76% (+41%) | 82% (+44%) |

## Skill Version(s): <br>
0.6.0-alpha.20260715 (source: git tag) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
