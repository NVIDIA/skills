## Description: <br>
Use for CUDA-Q setup, simulation targets, QPU access, and @cudaq.kernel authoring guidance. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill for CUDA-Q installation, GPU simulation target selection, QPU access setup, multi-GPU execution, and Python @cudaq.kernel authoring guidance. <br>

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
- [CUDA-Q Onboarding Reference](references/onboarding.md) <br>
- [CUDA-Q Authoring Reference](references/authoring.md) <br>
- [CUDA-Q Documentation](https://nvidia.github.io/cuda-quantum/latest/) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Configuration instructions] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
13 evaluation tasks (10 positive, 3 negative), 3 attempts per task in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against reference answers. <br>
- Discoverability: Checks whether the expected skill was selected and activated when needed. <br>
- Effectiveness: Checks whether the skill helped the agent complete the user's goal and follow expected workflow. <br>
- Efficiency: Checks tool-call productivity and token usage efficiency. <br>

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
| Overall | 94.9% | 88.2% |
| Security | 100.0% → 100.0% (±0.0 pts) | 84.6% → 92.3% (+7.7 pts) |
| Correctness | 87.7% → 100.0% (+12.3 pts) | 92.3% → 95.4% (+3.1 pts) |
| Discoverability | 97.5% | 93.0% |
| Effectiveness | 67.1% → 85.4% (+18.3 pts) | 73.9% → 84.8% (+10.9 pts) |
| Efficiency | 91.6% | 75.7% |

## Skill Version(s): <br>
1.1.1 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
