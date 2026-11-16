## Description: <br>
Guide an agent through GPU-accelerating a codebase: setup interview, flavor-specific translation (Fortran OpenACC, Fortran OpenMP, C++ CUDA, C++ Kokkos), and NVIDIA Nsight Systems (nsys) profile-guided orchestration. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and HPC engineers use this skill to port Fortran and C++ applications to NVIDIA GPUs through an iterative, profile-guided round loop with per-function numerical verification against the original code. <br>

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
- [Setup guide](references/setup-guide.md) <br>
- [Verification guide](references/verification-guide.md) <br>
- [Coverage set guide](references/coverage-set-guide.md) <br>
- [Skill feedback log](references/skill-feedback-log.md) <br>
- [Nsys orchestrator](orchestrators/nsys/orchestrator.md) <br>
- [NVIDIA Nsight Developer Tools](https://developer.nvidia.com/tools-overview) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Files, Shell commands, Analysis] <br>
**Output Format:** [Git commits of source changes, JSON state files, and Markdown iteration reports in chat] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-sonnet-5-5`) <br>
- Codex (`openai/openai/gpt-6-luna`) <br>



## Evaluation Tasks: <br>
21 evaluation tasks (18 positive, 3 negative), 1 attempt per task, each in an isolated sandbox. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill resists unsafe behavior such as prompt injection from inspected content. <br>
- Correctness: Whether the agent produces correct answers and follows required verification rules. <br>
- Discoverability: Whether the skill is discovered and activated when needed. <br>
- Effectiveness: Whether the skill helps the agent complete the user's goal and expected workflow. <br>
- Efficiency: Whether skill and tool usage, including tokens, is not wasted. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 53.3% → 60.0% (+6.7 points) | 46.7% → 74.3% (+27.6 points) |
| Effectiveness | 22.9% → 32.3% (+9.4 points) | 22.7% → 55.1% (+32.4 points) |
| Discoverability (with skill) | 9.9% | 75.8% |
| Efficiency (with skill) | 80.4% | 84.4% |

## Skill Version(s): <br>
6.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
