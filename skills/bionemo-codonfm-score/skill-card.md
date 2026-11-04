## Description: <br>
Validate, prepare, or run public CodonFM Encodon masked-codon variant scoring and review compatibility of its scoring workflows. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and computational biologists use this skill to validate variant CSV inputs and run or prepare masked-codon scoring commands with public Encodon models. <br>

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
- [NV CodonFM Encodon (NGC Catalog)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/nv_codonfm_encodon) <br>
- [NVIDIA Deep Bio Research](https://research.nvidia.com/labs/dbr) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Produces NumPy arrays (ref_likelihoods, alt_likelihoods, likelihood_ratios, ids) when inference executes] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
3 evaluation tasks (2 positive, 1 negative) in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal and followed the expected workflow behavior. <br>
- Efficiency: Checks tool-call productivity and token efficiency to detect wasted skill and tool usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 96.1% | 84.8% |
| Security | 100.0% → 100.0% (±0.0 pts) | 33.3% → 66.7% (+33.4 pts) |
| Correctness | 100.0% → 100.0% (±0.0 pts) | 100.0% → 100.0% (±0.0 pts) |
| Discoverability | 95.0% | 80.0% |
| Effectiveness | 90.0% → 93.3% (+3.3 pts) | 90.0% → 96.7% (+6.7 pts) |
| Efficiency | 92.1% | 80.7% |

## Skill Version(s): <br>
be43117 (source: git SHA, committed 2026-10-07) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
