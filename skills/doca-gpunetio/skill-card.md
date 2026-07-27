## Description: <br>
Use this skill when the user is doing hands-on DOCA GPUNetIO programming — wiring a CUDA kernel on an NVIDIA GPU to a doca-eth queue via doca_gpu_eth_rxq / doca_gpu_eth_txq, standing up the per-CUDA-device doca_gpu context, designing the persistent CUDA kernel that drains the GPU-visible queue, running the dual capability check (DOCA cap-query plus cudaGetDeviceProperties), registering cudaMalloc pools via doca_buf_arr_create_*, or debugging DOCA_ERROR_* returns from the GPUNetIO API. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and engineers building GPU-accelerated packet processing applications that wire NVIDIA DOCA network queues into persistent CUDA kernels for GPU-initiated networking on BlueField DPUs and ConnectX NICs. <br>

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
- [DOCA GPUNetIO Programming Guide](https://docs.nvidia.com/doca/sdk/DOCA-GPUNetIO/index.html) <br>
- [DOCA Samples (GitHub)](https://github.com/NVIDIA-DOCA/doca-samples) <br>
- [DOCA Platform Framework (GitHub)](https://github.com/NVIDIA/doca-platform) <br>
- [DOCA SDK Documentation](https://docs.nvidia.com/doca/sdk/index.html) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline C/CUDA and bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 4 internal skill evaluation tasks (3 positive activation, 1 negative activation). Pass threshold: 50%. Overall verdict: PASS. <br>

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
| Correctness | 4 | 100% (+20%) | 100% (+10%) |
| Discoverability | 4 | 100% (+25%) | 95% (+34%) |
| Effectiveness | 4 | 94% (+55%) | 100% (+65%) |
| Efficiency | 4 | 100% (+25%) | 95% (+56%) |

## Skill Version(s): <br>
d33a8af (source: git SHA, committed 2026-07-26) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
