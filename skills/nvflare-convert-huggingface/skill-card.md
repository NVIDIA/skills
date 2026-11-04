## Description: <br>
Convert existing Hugging Face Transformers Trainer or TRL SFTTrainer training code into an NVFLARE federated job using flare.patch(trainer), local validation, and job export; use when the user names Hugging Face or preliminary source inspection identifies one Hugging Face owner, and not for manual PyTorch loops, Lightning, inference-only pipelines, deployment, or experiment workflows. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers converting Hugging Face Transformers or TRL training code into NVFLARE federated learning jobs for secure, privacy-preserving distributed training. <br>

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
- [huggingface-conversion.md](references/huggingface-conversion.md) <br>
- [huggingface-detection.md](references/huggingface-detection.md) <br>
- [huggingface-state-and-distributed.md](references/huggingface-state-and-distributed.md) <br>
- [huggingface-validation.md](references/huggingface-validation.md) <br>
- [NVIDIA FLARE Documentation](https://nvflare.readthedocs.io/en/main) <br>
- [NVIDIA FLARE: Federated Learning from Simulation to Real-World](https://arxiv.org/abs/2210.13291) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions, Shell commands] <br>
**Output Format:** [Python source files and Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
21 evaluation tasks (21 positive) from a pinned dataset snapshot, one attempt per task, each run in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use, avoiding unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and followed the expected workflow. <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage. <br>

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
| Overall | 85.9% — baseline ran, but no comparable score was available; uplift unavailable | 76.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 45.2% → 100.0% (+54.8 points) | 57.1% → 73.8% (+16.7 points) |
| Correctness | 71.4% → 93.3% (+21.9 points) | 70.5% → 90.5% (+20.0 points) |
| Discoverability | 87.1% — baseline ran, but no comparable score was available; uplift unavailable | 75.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 47.4% → 69.1% (+21.7 points) | 50.3% → 66.9% (+16.6 points) |
| Efficiency | 79.8% — baseline ran, but no comparable score was available; uplift unavailable | 75.6% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
