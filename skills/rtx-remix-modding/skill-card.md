## Description: <br>
Mod or remaster a game with RTX Remix — open and edit projects, swap textures and models, and connect to the Remix Toolkit App via MCP. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and game modders use this skill to mod or remaster classic DirectX 8/9 games using NVIDIA RTX Remix, including opening projects, swapping texture and model assets, and managing USD layers through the Remix Toolkit's MCP interface. <br>

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
- [tool-mode.md](references/tool-mode.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline tool calls] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
10 evaluation tasks (7 positive, 3 negative), 3 attempts per task, evaluated in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against a reference answer. <br>
- Discoverability: Checks whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Checks whether the user's goal was achieved and expected workflow behavior was followed (equal-weight mean of goal_accuracy and behavior_check). <br>
- Efficiency: Checks tool-call productivity and token usage efficiency (50% tool productivity, 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (routing scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 90.4% — uplift unavailable | 80.2% — uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 87.5% → 83.3% (-4.2 points) |
| Correctness | 25.0% → 70.9% (+45.9 points) | 18.0% → 58.3% (+40.3 points) |
| Discoverability | 100.0% — uplift unavailable | 95.0% — uplift unavailable |
| Effectiveness | 32.1% → 82.3% (+50.2 points) | 34.0% → 69.0% (+35.0 points) |
| Efficiency | 98.7% — uplift unavailable | 95.1% — uplift unavailable |

## Skill Version(s): <br>
a836b2d41 (source: git SHA, committed 2026-09-22) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
