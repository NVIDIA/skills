## Description: <br>
Run and validate an end-to-end Mission Control showcase with a locally installed Isaac Sim launched in its GUI window, driven through the isaac-sim-remote Python server, with Nova Carter SIL. <br>

This skill is for demonstration purposes and not for production usage. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to run, replay, and validate the canonical small-warehouse Nova Carter showcase that drives Mission Control and a locally installed Isaac Sim together, and to diagnose the integrated scenario. <br>

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
- [Workflow stage router](references/workflow.md) <br>
- [Troubleshooting](references/troubleshooting.md) <br>
- [Publishing layout](references/publishing-layout.md) <br>
- [bring-up-cloud-stack adapter](references/bring-up-cloud-stack/README.md) <br>
- [change-map adapter](references/change-map/README.md) <br>
- [change-fleet-composition adapter](references/change-fleet-composition/README.md) <br>
- [isaac-sim-remote adapter](references/isaac-sim-remote/README.md) <br>
- [isaac-sim-installation adapter](references/isaac-sim-installation/README.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Files] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [The bundled runner writes run-manifest.json and the machine-readable run-result.json acceptance artifact into a temporary work directory.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-sonnet-5-5`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), one attempt per task, each run in its own isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct against the reference answer? <br>
- Discoverability: Was the right skill loaded and activated when needed, avoiding decoys? <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Token-usage productivity. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 84.8% — baseline ran, but no comparable score was available; uplift unavailable | 80.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 50.0% → 75.0% (+25.0 points) |
| Correctness | 60.0% → 95.0% (+35.0 points) | 45.0% → 95.0% (+50.0 points) |
| Discoverability | 86.7% — baseline ran, but no comparable score was available; uplift unavailable | 85.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 46.3% → 72.9% (+26.6 points) | 31.5% → 68.8% (+37.3 points) |
| Efficiency | 69.6% — baseline ran, but no comparable score was available; uplift unavailable | 77.0% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
