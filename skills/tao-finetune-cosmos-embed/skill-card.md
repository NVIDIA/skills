## Description: <br>
Cosmos-Embed1 video-text embedding for text-to-video retrieval, video-to-video search, semantic deduplication, and fine-tuning. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to fine-tune, evaluate, run inference on, and export Cosmos-Embed1 video-text embedding models for text-to-video retrieval, video-to-video search, and semantic deduplication. <br>

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
- [skill_info.yaml](references/skill_info.yaml) <br>
- [spec_template_train.yaml](references/spec_template_train.yaml) <br>
- [spec_template_evaluate.yaml](references/spec_template_evaluate.yaml) <br>
- [spec_template_inference.yaml](references/spec_template_inference.yaml) <br>
- [spec_template_export_onnx.yaml](references/spec_template_export_onnx.yaml) <br>
- [spec_template_export_hf.yaml](references/spec_template_export_hf.yaml) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 1 task (1 positive) in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the right skill was found and executed when needed. <br>
- Effectiveness: Checks whether the skill helped the agent complete the user's goal and expected workflow. <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Detects unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `skill_execution`: Verifies whether the expected skill was found and executed. <br>
- `goal_accuracy`: Verifies whether the user's goal was achieved. <br>
- `behavior_check`: Verifies whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Verifies routing quality, workspace-aware skill reads, and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 40% → 100% (+60 points) | 34% → 95% (+62 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 20% → 100% (+80 points) | 20% → 100% (+80 points) |
| Discoverability | 50% → 100% (+50 points) | 0% → 94% (+94 points) |
| Effectiveness | 17% → 100% (+83 points) | 48% → 83% (+35 points) |
| Efficiency | 12% → 100% (+88 points) | 0% → 100% (+100 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
