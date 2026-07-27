## Description: <br>
Use this skill when the user is doing hands-on DOCA Rivermax work on a BlueField DPU or ConnectX host — standing up doca_rmax_in_stream (receive) sessions for timing-precise media-over-IP (SMPTE ST 2110 video/audio, market data, scientific feeds), confirming the Rivermax SDK + license precondition before any DOCA-side code, running doca_rmax_get_*_supported capability queries, pairing with doca-eth queues and doca-flow steering, or debugging DOCA_ERROR_* from a Rivermax call. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
External developers building applications that consume the DOCA Rivermax integration — users whose code calls doca_rmax_* to drive timing-precise media-over-IP streams (SMPTE ST 2110, real-time market data, high-throughput scientific instrument streams) on a BlueField or ConnectX host. <br>

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
- [DOCA Developer Forum](https://forums.developer.nvidia.com/c/infrastructure/doca/370) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Shell commands, Code] <br>
**Output Format:** [Markdown with inline bash and C code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 4 internal evaluation tasks (3 positive skill-activation, 1 negative). <br>

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



## Evaluation Results: <br>
| Dimension | Num | Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) | Codex (`openai/openai/gpt-5.5`) |
|---|---:|---:|---:|
| Security | 4 | 100% (+0%) | 100% (+0%) |
| Correctness | 4 | 100% (+40%) | 100% (+35%) |
| Discoverability | 4 | 100% (+25%) | 95% (+45%) |
| Effectiveness | 4 | 88% (+78%) | 94% (+69%) |
| Efficiency | 4 | 90% (+26%) | 100% (+75%) |

## Skill Version(s): <br>
7b32072 (source: git SHA, committed 2026-07-26) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
