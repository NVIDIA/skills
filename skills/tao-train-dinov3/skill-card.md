## Description: <br>
DINOv3 continual self-supervised pre-training that domain-adapts public DINOv3 ViT backbones on unlabeled images via teacher-student self-distillation (DINO + iBOT + KoLeo, optional Gram anchoring) and converts the EMA teacher into a timm-format backbone for downstream tasks. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to continue self-supervised training of DINOv3 backbones on unlabeled domain images with NVIDIA TAO Toolkit containers, then convert, export, or run inference with selected teacher checkpoints. <br>

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
- [skill_info.yaml](references/skill_info.yaml) <br>
- [TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Files] <br>
**Output Format:** [Markdown with inline bash and YAML code blocks; YAML spec files] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Training writes full, teacher, and student checkpoints to the configured results directory.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
2 positive evaluation tasks (tao-train-dinov3-basic, tao-train-dinov3-checkpoint-selection), 1 attempt per task, each in an isolated k8s sandbox pod, compared against a no-skill baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the security signal. <br>
- Correctness: Is the answer correct? Scored from final-answer accuracy. <br>
- Discoverability: Was the right skill loaded when needed? Scored from skill execution. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal accuracy and behavior check. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight mean of tool productivity and token efficiency. <br>

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
| Overall | 93.3% — no comparable baseline; uplift unavailable | 95.7% — no comparable baseline; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 0.0% → 100.0% (+100.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 100.0% — no comparable baseline; uplift unavailable | 95.0% — no comparable baseline; uplift unavailable |
| Effectiveness | 12.5% → 93.8% (+81.3 points) | 70.0% → 85.0% (+15.0 points) |
| Efficiency | 73.0% — no comparable baseline; uplift unavailable | 98.5% — no comparable baseline; uplift unavailable |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
