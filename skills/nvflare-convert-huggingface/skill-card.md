## Description: <br>
Convert existing Hugging Face Transformers Trainer or TRL SFTTrainer training code into an NVFLARE federated job using flare.patch(trainer), local validation, and job export. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and ML engineers converting existing Hugging Face Transformers or TRL training scripts into federated learning jobs using NVIDIA FLARE. <br>

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
- [Hugging Face Conversion Reference](references/huggingface-conversion.md) <br>
- [Hugging Face Detection Reference](references/huggingface-detection.md) <br>
- [Hugging Face State and Distributed Reference](references/huggingface-state-and-distributed.md) <br>
- [Hugging Face Validation Reference](references/huggingface-validation.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Files, Shell commands] <br>
**Output Format:** [Python source files and NVFLARE job configuration] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
21 evaluation tasks (21 positive) in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Verifies final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Measures routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Codex (Baseline → Skill Uplift) |
|---|---:|
| Overall | 46% → 77% (+31 points) |
| Security | 45% → 60% (+14 points) |
| Correctness | 70% → 99% (+29 points) |
| Discoverability | 30% → 68% (+38 points) |
| Effectiveness | 49% → 82% (+32 points) |
| Efficiency | 33% → 77% (+44 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
