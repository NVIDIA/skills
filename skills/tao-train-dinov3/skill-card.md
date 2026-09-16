## Description: <br>
Domain-adapts public DINOv3 ViT backbones on unlabeled images via teacher-student self-distillation (DINO + iBOT + KoLeo, optional Gram anchoring) and converts the EMA teacher into a timm-format backbone for downstream tasks. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers who need to domain-adapt DINOv3 vision transformer backbones on unlabeled images for downstream computer vision tasks such as classification, detection, segmentation, or depth estimation. <br>

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
- [DINOv3 method and configuration](references/dinov3-method.md) <br>
- [DINOv3 tuning and evaluation](references/dinov3-recipes.md) <br>
- [Skill info](references/skill_info.yaml) <br>
- [Train spec template](references/spec_template_train.yaml) <br>
- [High-resolution train spec template](references/spec_template_train_highres.yaml) <br>
- [Convert spec template](references/spec_template_convert.yaml) <br>
- [Export spec template](references/spec_template_export.yaml) <br>
- [Inference spec template](references/spec_template_inference.yaml) <br>
- [TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
2 evaluation tasks (2 positive), 3 attempts per task, each in an isolated k8s-sandbox pod. Evaluator version 1.5.6. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Measures final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was selected, decoys avoided, and workflow executed. <br>
- Effectiveness: Equal-weight mean of goal completion (goal_accuracy) and expected workflow adherence (behavior_check). <br>
- Efficiency: 50% tool-call productivity (skill_efficiency) and 50% token efficiency. <br>

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
| Overall | 97.0% | 87.2% |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 3.3% → 100.0% (+96.7 points) | 80.0% → 100.0% (+20.0 points) |
| Discoverability | 100.0% | 47.5% |
| Effectiveness | 8.8% → 100.0% (+91.2 points) | 60.0% → 90.0% (+30.0 points) |
| Efficiency | 84.8% | 98.5% |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
