## Description: <br>
Find version-aware official NVSHMEM and NVSHMEM4Py documentation for releases, installation, APIs, runtime settings, transports, containers, and troubleshooting. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers working with NVSHMEM use this skill to find and retrieve version-specific official NVSHMEM and NVSHMEM4Py documentation covering releases, installation, APIs, runtime settings, transports, containers, and troubleshooting. <br>

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
- [NVSHMEM API Documentation (latest)](https://docs.nvidia.com/nvshmem/api/latest/) <br>
- [NVSHMEM Release Notes and Installation Guide](https://docs.nvidia.com/nvshmem/release-notes-install-guide/) <br>
- [topic-map.md](references/topic-map.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Documentation links] <br>
**Output Format:** [Markdown] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
8 evaluation tasks (7 positive, 1 negative), 3 attempts per task, each in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was selected and activated when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and followed the expected workflow. <br>
- Efficiency: Whether the skill avoided wasted tool calls and excessive token usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity scored under Efficiency. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 90.4% | 86.8% |
| Security | 90.0% → 100.0% (+10.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 58.0% → 100.0% (+42.0 points) | 97.5% → 100.0% (+2.5 points) |
| Discoverability | 80.0% | 70.0% |
| Effectiveness | 43.5% → 97.9% (+54.4 points) | 72.2% → 94.2% (+22.0 points) |
| Efficiency | 74.1% | 69.8% |

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
