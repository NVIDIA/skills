## Description: <br>
Run and validate an end-to-end Mission Control showcase with a locally installed Isaac Sim launched in its GUI window, driven through the isaac-sim-remote Python server, with Nova Carter SIL. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and engineers (NVIDIA Isaac Employees and External robotics developers) use this skill to run, replay, and diagnose an end-to-end Mission Control showcase that drives a Nova Carter robot through a canonical small-warehouse Isaac Sim scenario and validates mission completion. <br>

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
- [Stage router workflow](references/workflow.md) <br>
- [Troubleshooting and lifecycle states](references/troubleshooting.md) <br>
- [Publishing layout](references/publishing-layout.md) <br>
- [bring-up-cloud-stack adapter](references/bring-up-cloud-stack/README.md) <br>
- [change-map adapter](references/change-map/README.md) <br>
- [change-fleet-composition adapter](references/change-fleet-composition/README.md) <br>
- [isaac-sim-remote adapter](references/isaac-sim-remote/README.md) <br>
- [isaac-sim-installation adapter](references/isaac-sim-installation/README.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Files, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks, plus JSON runtime artifacts] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Writes run-manifest.json and run-result.json acceptance artifacts and run logs into a per-run work directory under the system temp location.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-sonnet-5-5`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), each attempt run in its own isolated sandbox pod on evaluator version 1.5.6. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct? Final-answer correctness against the reference answer. <br>
- Discoverability: Was the right skill loaded when needed, with decoys avoided and the workflow executed. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight mean of tool-call productivity and token efficiency. <br>

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
| Overall | 77.6% (baseline ran, no comparable score; uplift unavailable) | 84.8% (baseline ran, no comparable score; uplift unavailable) |
| Security | 100.0% → 75.0% (-25.0 points) | 75.0% → 87.5% (+12.5 points) |
| Correctness | 55.0% → 95.0% (+40.0 points) | 35.0% → 100.0% (+65.0 points) |
| Discoverability | 72.7% (baseline ran, no comparable score; uplift unavailable) | 85.3% (baseline ran, no comparable score; uplift unavailable) |
| Effectiveness | 40.6% → 71.9% (+31.3 points) | 31.5% → 73.8% (+42.3 points) |
| Efficiency | 73.3% (baseline ran, no comparable score; uplift unavailable) | 77.4% (baseline ran, no comparable score; uplift unavailable) |

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
