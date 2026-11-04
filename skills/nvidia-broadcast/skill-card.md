## Description: <br>
Use when controlling NVIDIA Broadcast through MCP to apply effects, process local media, select devices, or change camera resolution, or when Broadcast is missing or too old to expose the gateway and the user wants it installed; not for Broadcast app settings outside the MCP gateway. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to control NVIDIA Broadcast AI effects (camera, microphone, and speaker) through a local MCP gateway, including applying effects, processing local media files, selecting devices, changing camera resolution, and installing or updating NVIDIA Broadcast when needed. <br>

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
- [connection.md](references/connection.md) <br>
- [effects.md](references/effects.md) <br>
- [examples.md](references/examples.md) <br>
- [file-processing.md](references/file-processing.md) <br>
- [installation.md](references/installation.md) <br>
- [mcp-tool-contract.md](references/mcp-tool-contract.md) <br>
- [troubleshooting.md](references/troubleshooting.md) <br>
- [NVIDIA Broadcast](https://www.nvidia.com/broadcast-app/) <br>


## Skill Output: <br>
**Output Type(s):** [API Calls, Shell commands, Configuration instructions] <br>
**Output Format:** [MCP tool calls with JSON arguments, shell commands, and Markdown instructions] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
22 evaluation tasks (19 positive, 3 negative) run in isolated sandbox pods, with 1 attempt per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether the skill is safe to use, covering unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks whether the answer is correct against the reference answer. <br>
- Discoverability: Checks whether the right skill was loaded when needed, including skill execution and decoy avoidance. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal, combining goal completion (50%) and expected workflow adherence (50%). <br>
- Efficiency: Checks whether the skill avoided wasted tool calls and token usage, combining tool-call productivity (50%) and token efficiency (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Verifies absence of unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `skill_execution`: Verifies whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Verifies whether the user's goal was achieved. <br>
- `behavior_check`: Verifies whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Verifies tool-call productivity. <br>
- `token_efficiency`: Verifies actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 77.9% | 74.1% |
| Security | 100.0% → 100.0% (±0.0 pts) | 95.5% → 100.0% (+4.5 pts) |
| Correctness | 29.1% → 53.6% (+24.5 pts) | 30.0% → 49.1% (+19.1 pts) |
| Discoverability | 91.6% | 86.1% |
| Effectiveness | 39.2% → 49.1% (+9.9 pts) | 38.9% → 45.6% (+6.7 pts) |
| Efficiency | 95.0% | 89.5% |

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
