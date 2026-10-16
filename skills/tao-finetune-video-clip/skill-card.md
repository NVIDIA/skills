## Description: <br>
InternVideo2-CLIP L14 (TAO video_clip) for video-text retrieval, zero-shot classification, embedding extraction, LoRA fine-tuning, ONNX export, and TensorRT deployment. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers who need to fine-tune, evaluate, run inference on, export, and deploy InternVideo2-CLIP L14 models for video-text retrieval, zero-shot video classification, and embedding extraction using NVIDIA TAO Toolkit PyTorch and Deploy containers or a local virtualenv. <br>

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
- [tao-deploy-video-clip.md](references/tao-deploy-video-clip.md) <br>
- [tao-deploy-video-clip.skill_info.yaml](references/tao-deploy-video-clip.skill_info.yaml) <br>
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
3 evaluation tasks (3 positive), with 1 attempt per task, each run in an isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether the skill is safe to use (unsafe operations, secret leakage, and unauthorized access). <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the right skill was loaded when needed. <br>
- Effectiveness: Checks goal completion (50%) and expected workflow behavior adherence (50%). <br>
- Efficiency: Checks tool-call productivity (50%) and token efficiency (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 90.7% | 58.8% |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 0.0% → 100.0% (+100.0 points) | 26.7% → 60.0% (+33.3 points) |
| Discoverability | 80.0% | 0.0% |
| Effectiveness | 9.5% → 87.8% (+78.3 points) | 21.1% → 35.6% (+14.5 points) |
| Efficiency | 85.9% | 99.5% → 98.3% (-1.2 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
