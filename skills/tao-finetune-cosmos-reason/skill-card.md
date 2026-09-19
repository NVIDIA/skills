## Description: <br>
Shared Cosmos3 frontend that explicitly routes Cosmos Framework and Cosmos-RL, validates runtime model/video-dataset/SLURM inputs, consumes an SQSH or packaged backend image, optionally plans explicit clean source builds, prepares checkpoints, validates the first update in-process, and returns token-weighted losses and task-aware accuracy. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to fine-tune NVIDIA Cosmos3 vision-language models (Nano and Edge) on video-conversation and task-aware video-reasoning datasets, using either the Cosmos Framework or Cosmos-RL backend on Docker or SLURM platforms. <br>

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
- [TAO Skill Bank Repository](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>
- [Detailed Guide](references/detailed-guide.md) <br>
- [Cosmos Reason Parameters](references/cosmos-reason-parameters.md) <br>
- [Cosmos Reason Evaluate](references/cosmos-reason-evaluate.md) <br>
- [Cosmos Reason Launch](references/cosmos-reason-launch.md) <br>
- [Cosmos Backend Operations](references/cosmos-backend-operations.md) <br>
- [Cosmos Reproducibility Gates](references/cosmos-reproducibility-gates.md) <br>
- [Cosmos Data Specs](references/cosmos-data-specs.md) <br>
- [Skill Info](references/skill_info.yaml) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
7 evaluation tasks (7 positive), 3 attempts per task, each in an isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was selected and the workflow executed. <br>
- Effectiveness: Whether the skill helped complete the user's goal (50% goal completion + 50% expected workflow adherence). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage (50% tool-call productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (routing scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 86.1% — baseline ran, but no comparable score was available; uplift unavailable | 74.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 96.7% → 100.0% (+3.3 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 26.7% → 68.6% (+41.9 points) | 45.3% → 84.0% (+38.7 points) |
| Discoverability | 96.4% — baseline ran, but no comparable score was available; uplift unavailable | 35.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 20.7% → 80.0% (+59.3 points) | 26.9% → 52.6% (+25.7 points) |
| Efficiency | 85.6% — baseline ran, but no comparable score was available; uplift unavailable | 98.4% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
0.3.6 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
