## Description: <br>
Use for non-failing HoloHub app work with ./holohub: scaffold, build, run, test, visual evidence, lint, and flow benchmarking. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers building, testing, and shipping HoloHub applications through the ./holohub CLI workflow, from scaffolding and implementation to visual evidence collection and flow benchmarking. <br>

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
- [HoloHub application workflow](references/application-workflow.md) <br>
- [HoloHub flow benchmarking](references/flow-benchmarking.md) <br>
- [holohub CLI contract](references/holohub-cli-contract.md) <br>
- [NVIDIA HoloHub (GitHub)](https://github.com/nvidia-holoscan/holohub) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
5 evaluation tasks (3 positive, 2 negative), each in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and followed expected workflow behavior. <br>
- Efficiency: Whether the skill avoided wasted tool or skill usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 74% → 95% (+22 points) | 69% → 83% (+14 points) |
| Security | 90% → 100% (+10 points) | 100% → 100% (±0 points) |
| Correctness | 88% → 88% (±0 points) | 68% → 84% (+16 points) |
| Discoverability | 70% → 100% (+31 points) | 69% → 76% (+7 points) |
| Effectiveness | 57% → 89% (+32 points) | 55% → 74% (+19 points) |
| Efficiency | 63% → 100% (+37 points) | 51% → 80% (+29 points) |

## Skill Version(s): <br>
2aa6ab69 (source: git SHA, committed 2026-09-02) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
