## Description: <br>
Run the root-level workflow runtime policy or rule-based rollouts and verify simulator success. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to validate Isaac for Healthcare simulation workflows by running policy or rule-based rollouts and verifying simulator success for evaluation, checkpoints, or local controllers. <br>

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
**Output Type(s):** [Shell commands, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
20 evaluation tasks (20 positive), each run in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct? Measures final-answer correctness against the reference answer. <br>
- Discoverability: Was the right skill loaded when needed? Checks whether the expected skill was found and executed. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did it avoid wasted tool or skill usage? Measures routing quality, workspace-aware skill reads, and productive tool use. <br>

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
| Overall | 38% → 66% (+28 points) | Not available |
| Security | 95% → 40% (-55 points) | Not available |
| Correctness | 7% → 91% (+84 points) | Not available |
| Discoverability | 48% → 83% (+36 points) | Not available |
| Effectiveness | 4% → 31% (+27 points) | Not available |
| Efficiency | 37% → 86% (+50 points) | Not available |

## Skill Version(s): <br>
0.8.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
