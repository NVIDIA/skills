## Description: <br>
Use this skill when the user is doing hands-on NVMe-over-Fabrics storage-target work on a BlueField DPU or ConnectX NIC with DOCA STA — standing up a doca_sta DOCA Core context that accelerates the target-side NVMe-oF data path over RDMA, defining doca_sta_subsystem targets backed by local NVMe-PCI backend disks, checking device support via doca_sta_cap_is_supported, sizing per-connection I/O queues, or debugging DOCA_ERROR_* from a STA call. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
External developers and engineers building NVMe-over-Fabrics storage targets that consume DOCA STA on BlueField DPUs, using the skill for configuration, build, debug, and capability-discovery guidance. <br>

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
- [NVIDIA DOCA SDK Documentation](https://docs.nvidia.com/doca/sdk/index.html) <br>
- [DOCA Samples Repository](https://github.com/NVIDIA-DOCA/doca-samples) <br>
- [DOCA Platform Framework](https://github.com/NVIDIA/doca-platform) <br>
- [DOCA Developer Forum](https://forums.developer.nvidia.com/c/infrastructure/doca/370) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Shell commands, Code] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`claude-code`) <br>
- Codex (`codex`) <br>



## Evaluation Tasks: <br>
Evaluated against 8 recorded Tier 3 evaluation tasks using the NVSkills-Eval external profile. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies that the agent loaded the expected skill and workflow. <br>
- `skill_efficiency`: Checks routing quality, decoy avoidance, and redundant tool usage. <br>
- `accuracy`: Grades final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the overall user task completed successfully. <br>
- `behavior_check`: Verifies expected behavior steps, including safety expectations. <br>
- `token_efficiency`: Compares token usage with and without the skill. <br>



## Evaluation Results: <br>
| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 4 | 100% (+0%) | 100% (+0%) |
| Correctness | 4 | 100% (+68%) | 98% (+48%) |
| Discoverability | 4 | 96% (+71%) | 98% (+52%) |
| Effectiveness | 4 | 85% (+58%) | 98% (+59%) |
| Efficiency | 4 | 86% (+41%) | 95% (+38%) |

## Skill Version(s): <br>
59d6e8a (source: git SHA, committed 2026-07-15) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
