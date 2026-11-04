## Description: <br>
Validate coding-sequence CSVs, extract public CodonFM/Encodon embeddings, and choose checkpoints for downstream property modeling. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers working with codon sequences use this skill to validate CSV inputs, extract frozen Encodon embeddings, and select model checkpoints for downstream property modeling such as translation efficiency and mRNA stability prediction. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [No] <br>
**Credential Type(s):** [None] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Checkpoint Selection Guide](references/checkpoint-selection.md) <br>
- [NV-CodonFM-Encodon-1B-v1 (HuggingFace)](https://huggingface.co/nvidia/NV-CodonFM-Encodon-1B-v1) <br>
- [NV-CodonFM-Encodon-80M-v1 (HuggingFace)](https://huggingface.co/nvidia/NV-CodonFM-Encodon-80M-v1) <br>
- [NV-CodonFM-Encodon-600M-v1 (HuggingFace)](https://huggingface.co/nvidia/NV-CodonFM-Encodon-600M-v1) <br>
- [NV-CodonFM-Encodon-Cdwt-1B-v1 (HuggingFace)](https://huggingface.co/nvidia/NV-CodonFM-Encodon-Cdwt-1B-v1) <br>
- [CodonFM Encodon on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/nv_codonfm_encodon) <br>
- [CodonFM Preprint](https://research.nvidia.com/labs/dbr/assets/data/manuscripts/nv-codonfm-preprint.pdf) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks and JSON] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 4 evaluation tasks (4 positive) in isolated sandbox pods, using evaluator version 1.5.6. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use: checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill was selected and activated when needed, and decoys were avoided. <br>
- Effectiveness: Whether the skill helped complete the user's goal (50% goal accuracy + 50% expected workflow adherence). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage (50% tool-call productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity measured against expected tool-call patterns. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 96.1% | 93.4% |
| Security | 75.0% → 100.0% (+25.0 points) | 75.0% → 100.0% (+25.0 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 96.3% | 86.3% |
| Effectiveness | 85.6% → 90.6% (+5.0 points) | 98.8% → 93.8% (-5.0 points) |
| Efficiency | 93.6% | 86.9% |

## Skill Version(s): <br>
be43117 (source: git SHA, committed 2026-10-07) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
