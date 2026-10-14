## Description: <br>
Use this skill when the user is doing hands-on DOCA GPUNetIO programming — wiring a CUDA kernel on an NVIDIA GPU to a doca-eth queue via doca_gpu_eth_rxq / doca_gpu_eth_txq, standing up the per-CUDA-device doca_gpu context, designing the persistent CUDA kernel that drains the GPU-visible queue, running the dual capability check (DOCA cap-query plus cudaGetDeviceProperties), registering cudaMalloc pools via doca_buf_arr_create_*, or debugging DOCA_ERROR_* returns from the GPUNetIO API. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
External developers and engineers building applications on the DOCA GPUNetIO library use this skill to wire CUDA kernels on NVIDIA GPUs to DOCA Ethernet queues, stand up per-device GPU contexts, check GPU and DOCA capabilities, register CUDA buffers with DOCA, and debug GPUNetIO API errors. <br>

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
- [DOCA GPUNetIO Programming Guide](https://docs.nvidia.com/doca/sdk/DOCA-GPUNetIO/index.html) <br>
- [DOCA SDK Documentation](https://docs.nvidia.com/doca/sdk/index.html) <br>
- [DOCA Samples](https://github.com/NVIDIA-DOCA/doca-samples) <br>
- [CAPABILITIES.md](CAPABILITIES.md) <br>
- [TASKS.md](TASKS.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Code, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative) from skill-evaluator-dataset-snapshot/1, with 1 attempt per task, each run in its own isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether the skill is safe to use — unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks whether the final answer is correct against the reference answer. <br>
- Discoverability: Checks whether the right skill was loaded when needed. <br>
- Effectiveness: Checks whether the skill helped complete the task (50% goal completion + 50% expected workflow adherence). <br>
- Efficiency: Checks whether wasted tool calls and token usage were avoided (50% tool-call productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 92.5% — baseline ran, but no comparable score was available; uplift unavailable | 93.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 85.0% → 100.0% (+15.0 points) | 95.0% → 100.0% (+5.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 91.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 51.3% → 85.6% (+34.3 points) | 47.5% → 84.4% (+36.9 points) |
| Efficiency | 76.7% — baseline ran, but no comparable score was available; uplift unavailable | 90.5% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
96328e7 (source: git SHA, committed 2026-10-11) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
