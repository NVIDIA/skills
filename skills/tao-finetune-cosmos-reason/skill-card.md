## Description: <br>
Shared Cosmos3 frontend that explicitly routes Cosmos Framework and Cosmos-RL, validates runtime model/video-dataset/SLURM inputs, consumes an SQSH or packaged backend image, optionally plans explicit clean source builds, prepares checkpoints, validates the first update in-process, and returns token-weighted losses and task-aware accuracy. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers who fine-tune NVIDIA Cosmos3 vision-language models (Nano and Edge) on video conversation and task-aware video reasoning datasets using Docker or SLURM compute platforms. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key, Other [SSH key]] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Detailed Guide](references/detailed-guide.md) <br>
- [Cosmos Reason Evaluate](references/cosmos-reason-evaluate.md) <br>
- [Cosmos Reproducibility Gates](references/cosmos-reproducibility-gates.md) <br>
- [Cosmos Reason Parameters](references/cosmos-reason-parameters.md) <br>
- [Skill Info](references/skill_info.yaml) <br>
- [NVIDIA TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
7 evaluation tasks (7 positive), each with 3 attempts per task in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Goal completion (50%) combined with expected workflow adherence (50%). <br>
- Efficiency: Tool-call productivity (50%) combined with token efficiency (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 87.8% | 72.1% |
| Security | 94.1% → 100.0% (+5.9 pts) | 100.0% → 100.0% (±0.0 pts) |
| Correctness | 29.4% → 74.3% (+44.9 pts) | 46.7% → 77.8% (+31.1 pts) |
| Discoverability | 94.3% | 40.6% |
| Effectiveness | 17.8% → 85.4% (+67.6 pts) | 26.7% → 44.1% (+17.4 pts) |
| Efficiency | 84.9% | 98.0% |

## Skill Version(s): <br>
0.3.6 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
