## Description: <br>
Fine-tune any HuggingFace CV / VLM / LLM model on local NVIDIA GPUs inside an NGC PyTorch container when no dedicated TAO model skill matches. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers who need to fine-tune HuggingFace computer-vision, vision-language, or large-language models on local NVIDIA GPUs using NGC containers, when no dedicated TAO model skill covers the target model. <br>

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
- [NVIDIA TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>
- [NVIDIA Deep Learning Frameworks Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) <br>
- [Core Rules](references/core-rules.md) <br>
- [Detailed Workflow](references/detailed-workflow.md) <br>
- [Docker Runs](references/docker-runs.md) <br>
- [Hardware and Container Selection](references/hardware-container.md) <br>
- [Research Priorities](references/research-priorities.md) <br>
- [Error Playbook](references/error-playbook.md) <br>
- [CV Scripts](references/cv-scripts.md) <br>
- [VLM Scripts](references/vlm-scripts.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Code, Files] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (4 positive), each attempt in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was selected and the workflow executed. <br>
- Effectiveness: Checks goal completion and expected workflow adherence. <br>
- Efficiency: Checks tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (routing scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 80.2% | 58.7% |
| Security | 100.0% → 100.0% (±0.0 pp) | 100.0% → 100.0% (±0.0 pp) |
| Correctness | 25.7% → 85.0% (+59.3 pp) | 52.0% → 55.0% (+3.0 pp) |
| Discoverability | 58.8% | 0.0% |
| Effectiveness | 35.2% → 81.7% (+46.5 pp) | 45.0% → 39.1% (−5.9 pp) |
| Efficiency | 75.7% | 98.8% → 99.3% (+0.5 pp) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
