## Description: <br>
Cosmos3-Nano video QA supervised fine-tuning with FSDP parallelism across variable-width GPU configurations. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and ML engineers fine-tuning Cosmos3-Nano or compatible Cosmos Reason video QA models using SFT/LoRA, evaluating video question-answering performance, or running Cosmos-RL workflows. <br>

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
- [cosmos-reason-launch.md](references/cosmos-reason-launch.md) <br>
- [cosmos-reason-evaluate.md](references/cosmos-reason-evaluate.md) <br>
- [cosmos-reason-automl.md](references/cosmos-reason-automl.md) <br>
- [cosmos-reason-parameters.md](references/cosmos-reason-parameters.md) <br>
- [cosmos-reason-wts-gb300.md](references/cosmos-reason-wts-gb300.md) <br>
- [cosmos-data-specs.md](references/cosmos-data-specs.md) <br>
- [detailed-guide.md](references/detailed-guide.md) <br>
- [Cosmos3-Nano on Hugging Face](https://huggingface.co/nvidia/Cosmos3-Nano) <br>
- [NVIDIA TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
3 evaluation tasks (3 positive) run in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Checks whether the user's goal was achieved and expected workflow behavior was followed. <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Detects unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `skill_execution`: Verifies the expected skill was found and executed. <br>
- `goal_accuracy`: Verifies the user's goal was achieved. <br>
- `behavior_check`: Verifies the expected workflow behavior was followed. <br>
- `skill_efficiency`: Verifies routing quality and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 53% → 96% (+43 points) | 52% → 64% (+12 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 40% → 100% (+60 points) | 80% → 80% (±0 points) |
| Discoverability | 42% → 98% (+56 points) | 25% → 25% (±0 points) |
| Effectiveness | 38% → 93% (+55 points) | 52% → 81% (+28 points) |
| Efficiency | 45% → 92% (+46 points) | 0% → 33% (+33 points) |

## Skill Version(s): <br>
0.1.2 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
