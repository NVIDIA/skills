## Description: <br>
Guides hardware-offloaded Reed-Solomon erasure coding with the DOCA Erasure Coding library (doca_ec) on a BlueField DPU, ConnectX NIC, or host, covering parity generation, rebuilding lost blocks from the survivors, and refreshing parity after a change through the create / recover / update tasks. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
External developers and engineers building applications that consume the DOCA Erasure Coding C ABI (directly or through FFI/bindings) use this skill to offload Reed-Solomon erasure coding onto a BlueField DPU or ConnectX accelerator for distributed-storage resilience such as RAID-6-style block layouts, distributed file system parity, and object-storage erasure-coded buckets. <br>

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
- [DOCA Erasure Coding (SDK documentation)](https://docs.nvidia.com/doca/sdk/DOCA-Erasure-Coding/index.html) <br>
- [DOCA SDK documentation index](https://docs.nvidia.com/doca/sdk/index.html) <br>
- [DOCA samples (NVIDIA-DOCA/doca-samples)](https://github.com/NVIDIA-DOCA/doca-samples) <br>
- [CAPABILITIES.md](CAPABILITIES.md) <br>
- [TASKS.md](TASKS.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Code, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline code and command blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), each attempted once in its own isolated sandbox pod on a k8s-sandbox environment. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use — no unsafe operations, secret leakage, or unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill was loaded and activated when needed, avoiding decoys. <br>
- Effectiveness: Whether the skill helped complete the user's goal and the expected workflow was followed. <br>
- Efficiency: Whether the skill avoided wasted tool calls and excess token usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 98.2% (uplift unavailable) | 87.8% (uplift unavailable) |
| Security | 75.0% → 100.0% (+25.0 points) | 87.5% → 100.0% (+12.5 points) |
| Correctness | 45.0% → 100.0% (+55.0 points) | 70.0% → 90.0% (+20.0 points) |
| Discoverability | 100.0% (uplift unavailable) | 95.0% (uplift unavailable) |
| Effectiveness | 43.8% → 100.0% (+56.2 points) | 58.3% → 83.3% (+25.0 points) |
| Efficiency | 90.9% (uplift unavailable) | 70.9% (uplift unavailable) |

## Skill Version(s): <br>
c98e456 (source: git SHA, committed 2026-10-28) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
