## Description: <br>
Recognizes text content from cropped text-region images and supports CTC and attention-based decoders for training, evaluating, exporting, pruning, quantizing, retraining, and running inference for TAO OCRNet models. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers training, evaluating, pruning, quantizing, exporting, and running inference on OCRNet scene text recognition models using NVIDIA TAO Toolkit. <br>

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
- [TAO Deploy OCRNet](references/tao-deploy-ocrnet.md) <br>
- [Deploy skill_info.yaml](references/tao-deploy-ocrnet.skill_info.yaml) <br>
- [Spec Template: Train](references/spec_template_train.yaml) <br>
- [Spec Template: Evaluate](references/spec_template_evaluate.yaml) <br>
- [Spec Template: Export](references/spec_template_export.yaml) <br>
- [Spec Template: Inference](references/spec_template_inference.yaml) <br>
- [Spec Template: Prune](references/spec_template_prune.yaml) <br>
- [Spec Template: Quantize](references/spec_template_quantize.yaml) <br>
- [Spec Template: Retrain](references/spec_template_retrain.yaml) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
1 evaluation task (1 positive) from skill-evaluator-dataset-snapshot/1. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed. <br>
- Effectiveness: Checks whether the user's goal was achieved and expected workflow behavior was followed. <br>
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
| Overall | 46% → 93% (+47 points) | 55% → 95% (+40 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 20% → 100% (+80 points) | 100% → 100% (±0 points) |
| Discoverability | 50% → 100% (+50 points) | 0% → 94% (+94 points) |
| Effectiveness | 17% → 83% (+67 points) | 75% → 83% (+8 points) |
| Efficiency | 45% → 83% (+38 points) | 0% → 100% (+100 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
