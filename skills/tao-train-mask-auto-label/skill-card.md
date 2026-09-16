## Description: <br>
MAL (Mask Auto-Label) for weakly-supervised segmentation — produces segmentation masks from minimal annotations (point or box annotations) using a ViT-MAE backbone. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers training, evaluating, or running inference for weakly-supervised segmentation models using the NVIDIA TAO Toolkit. <br>

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
- [skill_info.yaml](references/skill_info.yaml) <br>
- [spec_template_train.yaml](references/spec_template_train.yaml) <br>
- [spec_template_evaluate.yaml](references/spec_template_evaluate.yaml) <br>
- [spec_template_inference.yaml](references/spec_template_inference.yaml) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
1 evaluation task (1 positive) against skill-evaluator-dataset-snapshot. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use (no unsafe operations, secret leakage, or unauthorized access). <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded and executed when needed. <br>
- Effectiveness: Whether the skill helped the agent complete the user's goal and expected workflow. <br>
- Efficiency: Whether the skill avoids wasted tool or skill usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies the expected skill was found and executed. <br>
- `skill_efficiency`: Measures routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the user's goal was achieved. <br>
- `behavior_check`: Checks whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 58% → 97% (+38 points) | 57% → 58% (+1 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 60% → 100% (+40 points) | 100% → 100% (±0 points) |
| Discoverability | 50% → 100% (+50 points) | 0% → 0% (±0 points) |
| Effectiveness | 53% → 100% (+47 points) | 88% → 92% (+5 points) |
| Efficiency | 29% → 83% (+55 points) | 0% → 0% (±0 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
