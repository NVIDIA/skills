## Description: <br>
Plan NVIDIA Parabricks pbrun commands, translate CPU options and check runtime readiness for individual tools, including planning-only questions. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and bioinformatics engineers use this skill to select the right NVIDIA Parabricks pbrun tool, generate version-aware v4.8.0 commands, translate CPU (for example STAR) options, check reference/index compatibility, and assess GPU, driver, Docker, and container runtime readiness for individual Parabricks tools. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Optional] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Tool Index](references/tool-index.md) <br>
- [Command Conventions](references/command-conventions.md) <br>
- [Runtime Environment](references/runtime-environment.md) <br>
- [Shared Options](references/shared-options.md) <br>
- [RNA Translation](references/rna-translation.md) <br>
- [Performance](references/performance.md) <br>
- [Parabricks Tool Reference](https://docs.nvidia.com/clara/parabricks/tool-reference) <br>
- [Output Accuracy and Compatible CPU Software Versions](https://docs.nvidia.com/clara/parabricks/about-parabricks/software-overview/output-accuracy-and-compatible-cpu-software-versions) <br>
- [Parabricks Getting Started](https://docs.nvidia.com/clara/parabricks/get-started) <br>
- [About Parabricks](https://docs.nvidia.com/clara/parabricks/about-parabricks) <br>
- [Parabricks 4.8.0 Release Notes](https://docs.nvidia.com/clara/parabricks/v4.8.0/about-parabricks/release-notes/4-8-0-1-release-notes) <br>
- [Parabricks Documentation MCP Server](https://docs.nvidia.com/clara/parabricks/_mcp/server) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis, Files] <br>
**Output Format:** [Markdown with inline bash code blocks; JSON from the bundled runtime-readiness and STAR translation helpers] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
25 evaluation tasks (24 positive, 1 negative), one attempt per task, each run in its own isolated sandbox pod and compared against a no-skill baseline. Evaluated 2026-10-27 with evaluator version 1.6.2; overall verdict PASS. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal. <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal. <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight mean of `skill_efficiency` and `token_efficiency`. <br>

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
| Overall | 96.3% — baseline ran, but no comparable score was available; uplift unavailable | 87.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 96.0% → 98.0% (+2.0 points) | 96.0% → 96.0% (±0.0 points) |
| Correctness | 75.2% → 100.0% (+24.8 points) | 95.2% → 99.2% (+4.0 points) |
| Discoverability | 99.6% — baseline ran, but no comparable score was available; uplift unavailable | 92.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 57.2% → 94.8% (+37.6 points) | 76.8% → 84.4% (+7.6 points) |
| Efficiency | 89.0% — baseline ran, but no comparable score was available; uplift unavailable | 63.3% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
1.4.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
