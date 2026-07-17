## Description: <br>
Use this skill when the user has an existing DPDK application and is adding DOCA capabilities in-place — most commonly DOCA Flow hardware steering — without rewriting the data-plane in DOCA-native form. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and engineers with existing DPDK-based packet-processing applications who want to add DOCA capabilities (hardware steering via DOCA Flow, accelerator integration) by linking the DOCA DPDK Bridge into the same process without rewriting their data-plane. <br>

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
- [DOCA SDK Documentation](https://docs.nvidia.com/doca/sdk/index.html) <br>
- [DOCA Samples (GitHub)](https://github.com/NVIDIA-DOCA/doca-samples) <br>
- [DOCA Platform Framework (GitHub)](https://github.com/NVIDIA/doca-platform) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions, Code] <br>
**Output Format:** [Markdown with inline bash and C code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`claude-code`) <br>
- Codex (`codex`) <br>



## Evaluation Tasks: <br>
Evaluated against 8 internal evaluation tasks using NVSkills-Eval Tier 3 profile (external). <br>

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
| Correctness | 4 | 100% (+55%) | 97% (+40%) |
| Discoverability | 4 | 98% (+73%) | 97% (+47%) |
| Effectiveness | 4 | 94% (+44%) | 97% (+52%) |
| Efficiency | 4 | 89% (+44%) | 94% (+42%) |

## Skill Version(s): <br>
c82da23 (source: git SHA, committed 2026-07-14) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
