## Description: <br>
Runs the DEFT embed-then-mine workflow for VCN AOI iterations — embeds the gap-analysis target parquet, embeds a source pool, and mines nearest-neighbour source images for downstream augmentation. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to mine nearest-neighbour source images from a pool for downstream training augmentation, as the step after gap analysis or routing in a Visual ChangeNet AOI pipeline. <br>

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
- [Setup](references/setup.md) <br>
- [Reference Invocation](references/reference-invocation.md) <br>
- [Outputs and Reporting](references/outputs-and-reporting.md) <br>
- [Troubleshooting](references/troubleshooting.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Files, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
1 evaluation task (1 positive), evaluated in isolated sandbox pods with Tier 3 live agent evaluation. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal and expected workflow (equal-weight mean of goal completion and behavior adherence). <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Verifies no unsafe operations, secret leakage, or unauthorized access occurred. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `skill_execution`: Verifies the expected skill was found and executed. <br>
- `goal_accuracy`: Verifies the user's goal was achieved. <br>
- `behavior_check`: Verifies the expected workflow behavior was followed. <br>
- `skill_efficiency`: Verifies routing quality, workspace-aware skill reads, and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 37% → 95% (+58 points) | 33% → 53% (+19 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 0% → 100% (+100 points) | 40% → 100% (+60 points) |
| Discoverability | 50% → 100% (+50 points) | 0% → 0% (±0 points) |
| Effectiveness | 0% → 100% (+100 points) | 27% → 63% (+37 points) |
| Efficiency | 36% → 75% (+39 points) | 0% → 0% (±0 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
