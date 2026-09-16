## Description: <br>
Extract per-molecule embeddings from any encoder-bearing KERMT checkpoint using containerized embedding extraction, writing per-readout .npy embeddings, canonical SMILES, and validity arrays to user-selected host directories. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and researchers use this skill to extract per-molecule embeddings from KERMT checkpoints for downstream molecular property prediction and cheminformatics tasks. <br>

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
- [Released KERMT Models](references/released-models.md) <br>
- [KERMT Paper (arXiv:2510.12719)](https://arxiv.org/abs/2510.12719) <br>
- [GROVER Paper (arXiv:2007.02835)](https://arxiv.org/abs/2007.02835) <br>
- [NV-KERMT-70M-v2 on Hugging Face](https://huggingface.co/nvidia/NV-KERMT-70M-v2) <br>


## Skill Output: <br>
**Output Type(s):** [Files, Shell commands] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
5 evaluation tasks (4 positive, 1 negative), 3 attempts per task, evaluated in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Whether the skill helps complete the user's goal (50% goal completion + 50% expected workflow adherence). <br>
- Efficiency: Whether the skill avoids wasted tool calls and token usage (50% tool-call productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity; routing is scored under Discoverability. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 82.0% | 83.3% |
| Security | 92.3% → 100.0% (+7.7 points) | 85.7% → 100.0% (+14.3 points) |
| Correctness | 24.6% → 84.0% (+59.4 points) | 80.0% → 84.0% (+4.0 points) |
| Discoverability | 93.8% | 86.3% |
| Effectiveness | 25.4% → 47.5% (+22.1 points) | 30.4% → 51.5% (+21.1 points) |
| Efficiency | 84.6% | 94.9% |

## Skill Version(s): <br>
77111e0 (source: git SHA, committed 2026-09-09) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
