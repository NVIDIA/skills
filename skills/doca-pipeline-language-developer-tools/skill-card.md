## Description: <br>
Guides an agent in driving the DOCA Pipeline Language (DPL) Developer Tools on a BlueField DPU — DPL Nspect for runtime state such as tables, keys, and counters, DPL Debugger for packet-flow visualization, DPL Admin for dynamic configuration changes in the DPL Runtime Daemon (`dpl_rtd`), P4 Runtime Shell as a P4Runtime control plane manager, and the DPL Runtime Controller SDK for custom P4Runtime applications and DPL shared memory APIs — while routing DPL program authoring/compilation/deployment, DOCA install, version, and hardware or firmware state changes to their owning skills or guides. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers, network and platform engineers, and AI agents use this skill to inspect and debug DPL programs loaded on a BlueField DPU — inspecting runtime tables and counters, visualizing packet flow, changing `dpl_rtd` configuration items, building custom P4Runtime control-plane applications, and diagnosing why the developer tools cannot reach `dpl_rtd`. <br>

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
- [CAPABILITIES.md](CAPABILITIES.md) <br>
- [TASKS.md](TASKS.md) <br>
- [DOCA SDK Documentation Index](https://docs.nvidia.com/doca/sdk/index.html) <br>
- [NVIDIA DOCA Developer Forum](https://forums.developer.nvidia.com/c/infrastructure/doca/370) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
5 evaluation tasks (3 positive, 2 negative), one attempt per task, each in an isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use, checking for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill was loaded when needed and decoys were avoided. <br>
- Effectiveness: Whether the skill helped complete the task, as the equal-weight mean of goal completion and expected-workflow adherence. <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage, as the equal-weight mean of tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline -> Skill Uplift) | Codex (Baseline -> Skill Uplift) |
|---|---:|---:|
| Overall | 92.2% (uplift unavailable) | 92.5% (uplift unavailable) |
| Security | 100.0% -> 100.0% (±0.0 points) | 90.0% -> 100.0% (+10.0 points) |
| Correctness | 40.0% -> 88.0% (+48.0 points) | 64.0% -> 96.0% (+32.0 points) |
| Discoverability | 100.0% (uplift unavailable) | 93.3% (uplift unavailable) |
| Effectiveness | 25.0% -> 85.0% (+60.0 points) | 35.0% -> 100.0% (+65.0 points) |
| Efficiency | 88.1% (uplift unavailable) | 73.4% (uplift unavailable) |

## Skill Version(s): <br>
c98e456 (source: git SHA, committed 2026-10-28) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
