## Description: <br>
Read-only GPU efficiency check for a Run:ai cluster that uses the Run:ai MCP server to classify projects, departments, and node pools holding GPUs as WELL_UTILIZED, UNDER_UTILIZED, or IDLE from average and p95 GPU utilization over one window (default 7 days), quantifying idle GPUs and idle GPU-hours and naming idle workloads. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers, platform engineers, and Run:ai cluster administrators use this skill to find which projects, departments, or node pools hold GPUs without actually computing on them, quantify idle GPUs and GPU-hours over a time window, and identify idle workloads, without taking any write action on the cluster. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [OAuth Token] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Utilization accounting (references/utilization-accounting.md)](references/utilization-accounting.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, API Calls] <br>
**Output Format:** [Markdown summary with a ranked utilization table] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Read-only; answers kept to one screen; reports sample coverage and marks holders without utilization data as UNKNOWN or LOW_DATA rather than estimating] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
22 evaluation tasks (16 positive, 6 negative), one attempt per task, each run in an isolated sandbox pod in a `k8s-sandbox` environment, compared against a no-skill baseline. <br>

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
| Overall | 91.6% — baseline ran, but no comparable score was available; uplift unavailable | 84.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 69.1% → 89.1% (+20.0 points) | 69.1% → 87.3% (+18.2 points) |
| Discoverability | 92.5% — baseline ran, but no comparable score was available; uplift unavailable | 89.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 58.9% → 81.4% (+22.5 points) | 61.8% → 85.3% (+23.5 points) |
| Efficiency | 94.8% — baseline ran, but no comparable score was available; uplift unavailable | 62.2% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
ae267e9e (source: git SHA, committed 2026-10-25) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
