## Description: <br>
Use for NVIDIA DGX Spark and related GB10 system identification, capacity, troubleshooting, known issues, updates, recovery, playbooks, examples, NVIDIA Sync, clustering, Custom Apps, Field Diagnostics, or support evidence; not for Apache Spark or unrelated meanings of spark. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers, engineers, and IT administrators use this skill to identify, troubleshoot, update, and maintain NVIDIA DGX Spark and related partner GB10 systems, including NVIDIA Sync usage, Field Diagnostics, and redacted support evidence preparation. <br>

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
- [DGX Spark hardware overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html) <br>
- [DGX Spark porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/) <br>
- [NVIDIA NGC DGX Spark collection](https://catalog.ngc.nvidia.com/orgs/nvidia/-/collections/dgx-spark/-/artifacts) <br>
- [DGX Spark release notes](https://docs.nvidia.com/dgx/dgx-spark/release-notes.html) <br>
- [DGX Spark Known Issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html) <br>
- [OS and component update guide](https://docs.nvidia.com/dgx/dgx-spark/os-and-component-update.html) <br>
- [DGX Spark catalog and playbooks](https://build.nvidia.com/spark) <br>
- [NVIDIA-Certified DGX Spark / GB10 systems](https://docs.nvidia.com/certification-programs/latest/nvidia-certified-systems.html#dgx-spark-grace-blackwell-gb10-systems) <br>
- [DGX Spark support options](https://docs.nvidia.com/dgx/dgx-spark/support.html) <br>
- [DGX Spark / GB10 developer forum](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-spark-gb10/719) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
22 evaluation tasks (20 positive, 2 negative) evaluated in isolated k8s-sandbox pods with 1 attempt per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use, checking for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed, including skill selection and decoy avoidance. <br>
- Effectiveness: Whether the skill helped complete the task, measured as the equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage, measured as the equal-weight mean of tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 94.5% — baseline ran, but no comparable score was available; uplift unavailable | 83.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 97.7% (-2.3 points) | 100.0% → 95.5% (-4.5 points) |
| Correctness | 76.4% → 97.3% (+20.9 points) | 77.3% → 88.2% (+10.9 points) |
| Discoverability | 99.8% — baseline ran, but no comparable score was available; uplift unavailable | 80.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 59.6% → 89.9% (+30.3 points) | 61.1% → 78.1% (+17.0 points) |
| Efficiency | 88.0% — baseline ran, but no comparable score was available; uplift unavailable | 75.8% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
0.1.5 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
