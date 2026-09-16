## Description: <br>
Grade or filter workflow HDF5 episodes with an OpenAI-compatible vision model. Use for visual success labels; do not use for replay, policy evaluation, or recordings without frames. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to grade or filter workflow HDF5 episode recordings using an OpenAI-compatible vision-language model, producing visual success labels for dataset annotation. <br>

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
2 evaluation tasks (2 positive) run in isolated k8s-sandbox pods, with 1 attempt per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether the skill is safe to use: no unsafe operations, secret leakage, or unauthorized access. <br>
- Correctness: Checks whether the final answer is correct against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal (goal completion and expected workflow adherence, equally weighted). <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Detects unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `skill_execution`: Verifies whether the expected skill was found and executed. <br>
- `goal_accuracy`: Verifies whether the user's goal was achieved. <br>
- `behavior_check`: Verifies whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Measures routing quality, workspace-aware skill reads, and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 47% → 85% (+38 points) | 17% → 63% (+46 points) |
| Security | 100% → 100% (±0 points) | 0% → 0% (±0 points) |
| Correctness | 30% → 100% (+70 points) | 10% → 100% (+90 points) |
| Discoverability | 50% → 94% (+44 points) | 28% → 81% (+53 points) |
| Effectiveness | 6% → 42% (+36 points) | 6% → 50% (+44 points) |
| Efficiency | 47% → 88% (+41 points) | 41% → 85% (+44 points) |

## Skill Version(s): <br>
0.8.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
