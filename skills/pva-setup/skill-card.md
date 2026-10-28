## Description: <br>
Install and configure the NVIDIA PVA SDK, NVVPU compiler, and L4T target runtime from the selected installation guide. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to prepare a PVA development environment, installing and configuring the PVA SDK, NVVPU compiler, and L4T target runtime on an x86-64 Ubuntu development host and an Arm64 Jetson Linux board based on Doctor's findings, then rerunning Doctor to confirm readiness. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [Other [SSH key]] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [PVA Solutions source repository](https://github.com/nvidia/pva-solutions.git) <br>
- [CI smoke evaluations (EVAL.md)](evals/EVAL.md) <br>
- [Skill benchmark results (BENCHMARK.md)](BENCHMARK.md) <br>


## Skill Output: <br>
**Output Type(s):** [Files, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown setup report with shell commands and exit codes] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
5 evaluation tasks (4 positive, 1 negative), one attempt each in isolated k8s-sandbox pods, covering missing installation sources, release mismatch, platform scope, the default PVA Solutions source plan, and negative triggering; the dataset does not verify package installation, compiler/EMU operation, or target runtime compatibility. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use, checking for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed and decoys were avoided. <br>
- Effectiveness: Whether the skill helped complete the user's goal and followed the expected workflow (equal-weight mean of goal accuracy and behavior check). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage (equal-weight mean of tool-call productivity and token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 96.4% — baseline ran, but no comparable score was available; uplift unavailable | 95.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 68.0% → 100.0% (+32.0 points) | 60.0% → 100.0% (+40.0 points) |
| Discoverability | 98.8% — baseline ran, but no comparable score was available; uplift unavailable | 90.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 63.7% → 100.0% (+36.3 points) | 63.7% → 100.0% (+36.3 points) |
| Efficiency | 83.4% — baseline ran, but no comparable score was available; uplift unavailable | 87.1% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
ef85bda (source: git SHA, committed 2026-10-28) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
