## Description: <br>
Fine-tune a manifest-backed GR00T or openpi remote Task on compatible LeRobot data. Use for training; do not use for inference-only Tasks or checkpoint rollout. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers fine-tuning GR00T or openpi policy models on LeRobot datasets for healthcare robotics workflows. <br>

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
- [Isaac for Healthcare Workflows Repository](https://github.com/isaac-for-healthcare/i4h-workflows) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
2 evaluation tasks (2 positive), each run in an isolated k8s-sandbox pod with 1 attempt per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use, checking for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and followed the expected workflow, scored as equal-weight mean of goal completion and behavior adherence. <br>
- Efficiency: Whether the skill avoided wasted tool or skill usage, measuring routing quality and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies the expected skill was found and executed. <br>
- `skill_efficiency`: Measures routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Assesses final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Determines whether the user's goal was achieved. <br>
- `behavior_check`: Verifies the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 38% → 78% (+39 points) | 59% → 67% (+7 points) |
| Security | 100% → 100% (±0 points) | 100% → 50% (-50 points) |
| Correctness | 0% → 100% (+100 points) | 60% → 100% (+40 points) |
| Discoverability | 47% → 94% (+47 points) | 53% → 78% (+25 points) |
| Effectiveness | 5% → 12% (+8 points) | 22% → 28% (+5 points) |
| Efficiency | 41% → 82% (+41 points) | 61% → 79% (+17 points) |

## Skill Version(s): <br>
0.8.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
