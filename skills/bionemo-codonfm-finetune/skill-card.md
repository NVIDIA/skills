## Description: <br>
Fine-tune public CodonFM Encodon checkpoints on labeled coding-sequence or coding-variant data using LoRA, head-only, or full fine-tuning. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and computational biologists use this skill to fine-tune NVIDIA CodonFM Encodon foundation models on labeled coding-sequence or coding-variant datasets for regression or classification tasks. <br>

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
- [NV-CodonFM-Encodon-80M-v1 (Hugging Face)](https://huggingface.co/nvidia/NV-CodonFM-Encodon-80M-v1) <br>
- [NV-CodonFM-Encodon-600M-v1 (Hugging Face)](https://huggingface.co/nvidia/NV-CodonFM-Encodon-600M-v1) <br>
- [NV-CodonFM-Encodon-1B-v1 (Hugging Face)](https://huggingface.co/nvidia/NV-CodonFM-Encodon-1B-v1) <br>
- [NV-CodonFM-Encodon on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/nv_codonfm_encodon) <br>
- [CenikLab TE Classic ML Data](https://github.com/CenikLab/TE_classic_ML/tree/main/data) <br>
- [NVIDIA Deep Bio Research](https://research.nvidia.com/labs/dbr) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Files] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 3 positive evaluation tasks in isolated k8s-sandbox pods, covering LoRA fine-tuning preparation, variant classification, and data preparation workflows. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use: checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed: skill selection, decoy avoidance, and workflow execution. <br>
- Effectiveness: Whether the skill helped complete the task: goal completion (50%) and expected workflow adherence (50%). <br>
- Efficiency: Whether wasted tool calls and token usage were avoided: tool-call productivity (50%) and token efficiency (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 85.0% | 82.8% |
| Security | 100.0% → 66.7% (-33.3 points) | 33.3% → 66.7% (+33.4 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 91.7% | 76.7% |
| Effectiveness | 86.7% → 86.7% (±0.0 points) | 96.7% → 90.0% (-6.7 points) |
| Efficiency | 80.0% | 80.7% |

## Skill Version(s): <br>
be43117 (source: git SHA, committed 2026-10-07) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
