## Description: <br>
Finetune a pretrained KERMT encoder on a labeled CSV. Validate the checkpoint and data, prepare features, and run containerized training. Use a local checkpoint or optionally download a pinned Hugging Face model bundle using HF_TOKEN if configured. Write model bundles, prepared data, logs, and trained models to user-selected host directories. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers finetuning pretrained KERMT molecular property prediction models on user-supplied labeled CSV datasets for small molecule drug property prediction tasks. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Optional] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Released Models](references/released-models.md) <br>
- [Multitask finetuning and acceleration of chemical pretrained models (KERMT paper)](https://arxiv.org/abs/2510.12719) <br>
- [Self-Supervised Graph Transformer on Large-Scale Molecular Data (GROVER paper)](https://arxiv.org/abs/2007.02835) <br>
- [NV-KERMT-70M-v2 on Hugging Face](https://huggingface.co/nvidia/NV-KERMT-70M-v2) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Files] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Produces run.json manifest, finetune logs, TensorBoard events, best-val and last checkpoints, and held-out test predictions] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
5 evaluation tasks (4 positive, 1 negative) with 3 attempts each, run in isolated sandbox pods (evaluator v1.5.6). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was selected and the workflow executed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal and expected workflow (50% goal_accuracy + 50% behavior_check). <br>
- Efficiency: Checks tool-call productivity and token usage (50% skill_efficiency + 50% token_efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 84.9% | 69.5% |
| Security | 88.5% → 100.0% (+11.5 points) | 27.3% → 71.4% (+44.1 points) |
| Correctness | 23.1% → 96.0% (+72.9 points) | 78.2% → 91.4% (+13.2 points) |
| Discoverability | 97.5% | 79.2% |
| Effectiveness | 18.9% → 51.0% (+32.1 points) | 30.7% → 35.7% (+5.0 points) |
| Efficiency | 79.8% | 69.7% |

## Skill Version(s): <br>
77111e0 (source: git SHA, committed 2026-09-09) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
