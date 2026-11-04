## Description: <br>
Operates NVIDIA App through its local MCP server to manage drivers, optimize games, configure laptop features, and control selected In-Game Overlay workflows. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to operate NVIDIA App for driver status checks, application management, per-game graphics optimization, laptop feature configuration, and In-Game Overlay capture and recording workflows. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [NVIDIA App Product Page](https://www.nvidia.com/en-us/software/nvidia-app/) <br>
- [connection.md](references/connection.md) <br>
- [general-tools.md](references/general-tools.md) <br>
- [overlay-capture.md](references/overlay-capture.md) <br>
- [overlay-state.md](references/overlay-state.md) <br>
- [mcp-tool-contract.md](references/mcp-tool-contract.md) <br>


## Skill Output: <br>
**Output Type(s):** [API Calls, Configuration instructions] <br>
**Output Format:** [Markdown] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
32 evaluation tasks (29 positive, 3 negative) from a pinned dataset snapshot, each executed in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the right skill was loaded and activated when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal and expected workflow (equal-weight mean of goal completion and workflow adherence). <br>
- Efficiency: Checks whether the skill avoided wasted tool calls and token usage (50% tool-call productivity, 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (routing scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 77.9% | 70.2% |
| Security | 96.9% → 68.8% (-28.1 points) | 85.9% → 76.6% (-9.3 points) |
| Correctness | 15.6% → 90.0% (+74.4 points) | 20.6% → 61.3% (+40.7 points) |
| Discoverability | 97.9% | 83.6% |
| Effectiveness | 29.7% → 37.0% (+7.3 points) | 30.2% → 35.7% (+5.5 points) |
| Efficiency | 95.9% | 93.9% |

## Skill Version(s): <br>
1.1.8 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
