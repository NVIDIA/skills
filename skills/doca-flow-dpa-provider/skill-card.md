## Description: <br>
Guides hands-on DOCA Flow DPA Provider work — exporting a `doca-flow` pipe or external resource (index-selector/memory) into BlueField DPA address space so a DPACC-built kernel can read counters, mutate hash-pipe entries, and update or read memory or index-selector resources inline with Flow. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
External developers and engineers building applications that consume the DOCA Flow DPA Provider library use this skill to wire an existing host-side doca-flow pipe and host-side doca-dpa execution context together, exporting the pipe or an external resource into the DPA address space so a DPA kernel can read counters, mutate entries, and read or write resources inline. It also covers the order-sensitive export handshake, the three DPA-side queue types, the DPA-side device API, and disambiguating DOCA_ERROR_* failures across the host, DPA, and export layers. <br>

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
- [Capabilities companion (CAPABILITIES.md)](CAPABILITIES.md) <br>
- [Tasks companion (TASKS.md)](TASKS.md) <br>
- [DOCA Flow SDK Guide](https://docs.nvidia.com/doca/sdk/doca-flow/index.html) <br>
- [DOCA DPA SDK Guide](https://docs.nvidia.com/doca/sdk/doca-dpa/index.html) <br>
- [DOCA Samples](https://github.com/NVIDIA-DOCA/doca-samples) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-sonnet-5-5`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative) drawn from the skill-evaluator dataset snapshot; 1 attempt per task in an isolated k8s-sandbox pod per attempt. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use — checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed and decoys were avoided. <br>
- Effectiveness: Whether the skill helped complete the user's goal and the expected workflow was followed (equal-weight mean of goal completion and behavior check). <br>
- Efficiency: Whether wasted tool calls and token usage were avoided (tool-call productivity plus token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | Not available | 97.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 100.0% → 100.0% (±0.0 points) |
| Correctness | Not available | 100.0% → 95.0% (-5.0 points) |
| Discoverability | Not available | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 63.1% → 94.8% (+31.7 points) |
| Efficiency | Not available | 100.0% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
3bc1c5d (source: git SHA, committed 2026-10-22) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
