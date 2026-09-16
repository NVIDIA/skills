## Description: <br>
Install cuOpt for Python, C, or server via pip, conda, or Docker; verify the install. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers installing NVIDIA cuOpt (GPU-accelerated optimization engine) for Python, C, or REST server deployment and verifying the installation. <br>

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
- [Verification Examples](references/verification_examples.md) <br>
- [cuOpt User Guide](https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html) <br>
- [cuOpt GitHub](https://github.com/NVIDIA/cuopt) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
7 evaluation tasks (7 positive) in isolated k8s-sandbox pods, evaluated 2026-08-12. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Verifies final-answer correctness against a reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Measures goal completion and expected workflow adherence. <br>
- Efficiency: Evaluates routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Detects unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies whether the expected skill was found and executed. <br>
- `skill_efficiency`: Measures routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Checks final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Determines whether the user's goal was achieved. <br>
- `behavior_check`: Verifies whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 56% → 90% (+34 points) | 63% → 91% (+28 points) |
| Security | 93% → 86% (-7 points) | 100% → 100% (±0 points) |
| Correctness | 86% → 94% (+9 points) | 89% → 100% (+11 points) |
| Discoverability | 27% → 96% (+69 points) | 47% → 94% (+46 points) |
| Effectiveness | 64% → 81% (+18 points) | 76% → 93% (+17 points) |
| Efficiency | 12% → 93% (+81 points) | 3% → 67% (+63 points) |

## Skill Version(s): <br>
26.10.00 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
