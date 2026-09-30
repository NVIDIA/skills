## Description: <br>
Run and validate an end-to-end Mission Control showcase with a locally installed Isaac Sim launched in its GUI window, driven through the isaac-sim-remote Python server, with Nova Carter SIL. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and robotics engineers running end-to-end NVIDIA Mission Control showcases with Isaac Sim and Nova Carter SIL for demos, showcase replays, simulated robot driving, and integration diagnostics. <br>

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
- [Workflow — Stage Router](references/workflow.md) <br>
- [Troubleshooting](references/troubleshooting.md) <br>
- [Bring Up Cloud Stack](references/bring-up-cloud-stack/README.md) <br>
- [Change Fleet Composition](references/change-fleet-composition/README.md) <br>
- [Change Map](references/change-map/README.md) <br>
- [Isaac Sim Remote](references/isaac-sim-remote/README.md) <br>
- [Isaac Sim Installation](references/isaac-sim-installation/README.md) <br>
- [Publishing Layout](references/publishing-layout.md) <br>
- [NVIDIA Agent Skills Documentation](https://docs.nvidia.com/skills) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative) executed in isolated sandbox pods with 3 attempts per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Whether the user's goal was achieved (50%) and expected workflow behavior was followed (50%). <br>
- Efficiency: Tool-call productivity (50%) and actual uncached token usage (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 89.9% | 86.9% |
| Security | 75.0% → 100.0% (+25.0 points) | 70.0% → 100.0% (+30.0 points) |
| Correctness | 27.5% → 85.0% (+57.5 points) | 32.0% → 95.0% (+63.0 points) |
| Discoverability | 93.3% | 78.0% |
| Effectiveness | 33.4% → 87.5% (+54.1 points) | 29.0% → 80.0% (+51.0 points) |
| Efficiency | 83.4% | 81.7% |

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
