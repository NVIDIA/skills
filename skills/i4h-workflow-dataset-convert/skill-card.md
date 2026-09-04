## Description: <br>
Convert workflow HDF5 recordings to LeRobot datasets for training or browser inspection. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to convert workflow HDF5 recordings into LeRobot datasets for robot policy training or browser-based data inspection. <br>

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
- [Isaac for Healthcare Workflows](https://github.com/isaac-for-healthcare/i4h-workflows) <br>
- [Agent Skills Specification](https://agentskills.io/specification) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Files, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 3 positive evaluation tasks, each run in an isolated sandbox pod with 1 attempt per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed. <br>
- Effectiveness: Checks whether the user's goal was achieved and the expected workflow behavior was followed. <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

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
| Overall | 51% → 82% (+30 points) | 46% → 72% (+25 points) |
| Security | 100% → 100% (±0 points) | 100% → 67% (-33 points) |
| Correctness | 53% → 80% (+27 points) | 53% → 80% (+27 points) |
| Discoverability | 48% → 96% (+48 points) | 35% → 85% (+50 points) |
| Effectiveness | 14% → 42% (+27 points) | 3% → 42% (+39 points) |
| Efficiency | 40% → 91% (+50 points) | 40% → 85% (+44 points) |

## Skill Version(s): <br>
0.8.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
