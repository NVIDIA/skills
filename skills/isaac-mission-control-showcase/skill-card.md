## Description: <br>
Run and validate an end-to-end Mission Control showcase with a locally installed Isaac Sim launched in its GUI window, driven through the isaac-sim-remote Python server, with Nova Carter SIL. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers running and validating end-to-end Mission Control showcases with Isaac Sim and Nova Carter SIL for demos, showcase replays, driving a simulated robot, or diagnosing the integrated small-warehouse scenario. <br>

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
- [Workflow](references/workflow.md) <br>
- [Troubleshooting](references/troubleshooting.md) <br>
- [Bring Up Cloud Stack](references/bring-up-cloud-stack/README.md) <br>
- [Change Fleet Composition](references/change-fleet-composition/README.md) <br>
- [Change Map](references/change-map/README.md) <br>
- [Isaac Sim Remote](references/isaac-sim-remote/README.md) <br>
- [Isaac Sim Installation](references/isaac-sim-installation/README.md) <br>
- [Publishing Layout](references/publishing-layout.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Files] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Machine-readable acceptance artifacts (run-manifest.json, run-result.json)] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), each with 3 attempts per task in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Whether the skill helped complete the user's goal (50% goal accuracy + 50% behavior check). <br>
- Efficiency: Tool-call productivity and token efficiency (50% each). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 95.4% | 83.7% |
| Security | 100.0% → 100.0% (±0.0 points) | 43.8% → 87.5% (+43.7 points) |
| Correctness | 37.5% → 100.0% (+62.5 points) | 40.0% → 80.0% (+40.0 points) |
| Discoverability | 99.3% | 86.7% |
| Effectiveness | 32.5% → 91.9% (+59.4 points) | 40.6% → 76.9% (+36.3 points) |
| Efficiency | 86.0% | 87.7% |

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
