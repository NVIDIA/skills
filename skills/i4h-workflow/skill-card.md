## Description: <br>
Orient users to the i4h workflow runtime and route them to the correct stage skill. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers working with the Isaac for Healthcare (i4h) workflow runtime use this skill to understand architecture, get support, and route to the correct stage skill for setup, data collection, training, or validation tasks. <br>

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
- [Workflow Ownership Map](references/repo-map.md) <br>
- [i4h-workflows Repository](https://github.com/isaac-for-healthcare/i4h-workflows) <br>
- [Agent Skills Specification](https://agentskills.io/specification) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Configuration instructions] <br>
**Output Format:** [Markdown] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
2 evaluation tasks (1 positive, 1 negative) run in isolated k8s-sandbox pods with 1 attempt per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use — checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and followed the expected workflow (equal-weight mean of goal completion and behavior adherence). <br>
- Efficiency: Whether the skill avoided wasted tool or skill usage — measures routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies whether the expected skill was found and executed. <br>
- `skill_efficiency`: Measures routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Evaluates final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Assesses whether the user's goal was achieved. <br>
- `behavior_check`: Verifies whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 61% → 73% (+12 points) | 71% → 68% (-3 points) |
| Security | 75% → 75% (±0 points) | 100% → 50% (-50 points) |
| Correctness | 50% → 50% (±0 points) | 80% → 60% (-20 points) |
| Discoverability | 72% → 97% (+25 points) | 72% → 94% (+22 points) |
| Effectiveness | 38% → 41% (+3 points) | 38% → 43% (+5 points) |
| Efficiency | 69% → 100% (+31 points) | 65% → 94% (+29 points) |

## Skill Version(s): <br>
0.8.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
