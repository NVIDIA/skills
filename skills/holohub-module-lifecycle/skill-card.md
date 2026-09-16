## Description: <br>
Use for reusable Holoscan Module work with ./holohub: scaffold, tests, editable install, DEB/WHEEL packaging, and clean-consumer proof. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers building reusable Holoscan Modules use this skill to scaffold, test, package (DEB/WHEEL), and prove clean-consumer installs through the HoloHub CLI. <br>

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
- [./holohub CLI contract](references/holohub-cli-contract.md) <br>
- [Module development through ./holohub](references/module-development.md) <br>
- [Module consumer iteration and packaging](references/module-consumer-packaging.md) <br>
- [HoloHub repository](https://github.com/nvidia-holoscan/holohub) <br>
- [Create a module tutorial](https://github.com/nvidia-holoscan/holohub/tree/main/tutorials/holoscan-modules/create-a-module) <br>
- [Use a module tutorial](https://github.com/nvidia-holoscan/holohub/tree/main/tutorials/holoscan-modules/use-a-module) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Code, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), each run in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use — checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the skill produces correct answers against the reference. <br>
- Discoverability: Whether the right skill was loaded and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and expected workflow (equal-weight mean of goal completion and behavior adherence). <br>
- Efficiency: Whether the skill avoided wasted tool or skill usage — routing quality and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 52% → 93% (+41 points) | 56% → 76% (+20 points) |
| Security | 75% → 100% (+25 points) | 100% → 75% (-25 points) |
| Correctness | 35% → 100% (+65 points) | 55% → 95% (+40 points) |
| Discoverability | 61% → 100% (+39 points) | 48% → 83% (+34 points) |
| Effectiveness | 34% → 66% (+33 points) | 45% → 52% (+7 points) |
| Efficiency | 55% → 97% (+42 points) | 34% → 75% (+41 points) |

## Skill Version(s): <br>
0504f91f (source: git SHA, committed 2026-09-02) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
