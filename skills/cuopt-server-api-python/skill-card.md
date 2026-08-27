## Description: <br>
cuOpt REST server — start server, endpoints, Python/curl client examples. Use when the user is deploying or calling the REST API. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers deploying the NVIDIA cuOpt REST server or writing Python/curl clients against its optimization endpoints for routing, LP, and MILP problems. <br>

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
- [cuOpt User Guide](https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html) <br>
- [cuOpt Docker Hub](https://hub.docker.com/r/nvidia/cuopt) <br>
- [cuOpt Examples](https://github.com/NVIDIA/cuopt-examples) <br>
- [assets/README.md](assets/README.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, API Calls, Code, Configuration instructions] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
8 evaluation tasks (8 positive) from skill-evaluator-dataset-snapshot. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Verifies final-answer correctness against reference answers. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and expected workflow. <br>
- Efficiency: Routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Detects unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 64% → 98% (+35 points) | 62% → 96% (+34 points) |
| Security | 100% → 100% (±0 points) | 69% → 100% (+31 points) |
| Correctness | 75% → 100% (+25 points) | 88% → 98% (+10 points) |
| Discoverability | 44% → 100% (+56 points) | 45% → 93% (+48 points) |
| Effectiveness | 56% → 95% (+38 points) | 72% → 88% (+15 points) |
| Efficiency | 44% → 97% (+53 points) | 37% → 100% (+63 points) |

## Skill Version(s): <br>
26.10.00 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
