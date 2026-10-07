## Description: <br>
Mod or remaster a classic DirectX 8/9 game with RTX Remix — open and edit projects, swap textures and models, and connect to the Remix Toolkit App via MCP. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and game modders use this skill to mod or remaster classic DirectX 8/9 games with NVIDIA RTX Remix, managing projects, swapping texture and model assets bound to captured prims, and editing USD layers through a local MCP connection to the Remix Toolkit. <br>

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
- [tool-mode.md](references/tool-mode.md) <br>


## Skill Output: <br>
**Output Type(s):** [API Calls, Analysis, Configuration instructions] <br>
**Output Format:** [Markdown] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
11 evaluation tasks (8 positive, 3 negative) in isolated sandbox pods, checking safety, correctness, discoverability, effectiveness, and efficiency. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed and decoys were avoided. <br>
- Effectiveness: Whether the skill helped complete the user's goal and followed the expected workflow (equal-weight mean of goal completion and behavior adherence). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage (50% tool-call productivity, 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (routing scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 86.2% | 87.0% |
| Security | 100.0% → 100.0% (±0.0 points) | 95.5% → 90.9% (-4.6 points) |
| Correctness | 29.1% → 65.5% (+36.4 points) | 16.4% → 72.7% (+56.3 points) |
| Discoverability | 92.5% | 95.0% |
| Effectiveness | 53.9% → 75.9% (+22.0 points) | 43.3% → 79.1% (+35.8 points) |
| Efficiency | 97.2% | 97.4% |

## Skill Version(s): <br>
9706a252a (source: git SHA, committed 2026-10-06) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
