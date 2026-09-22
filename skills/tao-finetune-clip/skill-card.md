## Description: <br>
CLIP vision-language model for image-text retrieval, zero-shot classification, embedding extraction, ONNX export, and TensorRT deployment. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to fine-tune, evaluate, export, and deploy NVIDIA TAO CLIP vision-language models for image-text retrieval, zero-shot classification, and embedding extraction. <br>

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
- [TAO Skill Bank Repository](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>
- [tao-deploy-clip.md](references/tao-deploy-clip.md) <br>
- [error-patterns.md](references/error-patterns.md) <br>
- [spec-param-inference.md](references/spec-param-inference.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
1 evaluation task (1 positive), 3 attempts per task, run in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was selected and the workflow executed. <br>
- Effectiveness: Checks whether the user's goal was achieved and expected workflow behavior was followed. <br>
- Efficiency: Checks tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Detects unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies expected skill selection and workflow execution. <br>
- `skill_efficiency`: Measures tool-call productivity. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Verifies whether the user's goal was achieved. <br>
- `behavior_check`: Verifies expected workflow behavior was followed. <br>
- `token_efficiency`: Measures actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 96.2% | 95.3% |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 6.7% → 100.0% (+93.3 points) | 20.0% → 100.0% (+80.0 points) |
| Discoverability | 100.0% | 95.0% |
| Effectiveness | 16.7% → 83.3% (+66.6 points) | 33.3% → 83.3% (+50.0 points) |
| Efficiency | 97.4% | 98.3% |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
