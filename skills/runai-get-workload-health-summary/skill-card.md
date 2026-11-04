## Description: <br>
Summarizes scoped Run:ai workload health or one workload. Not for root cause, quota, cluster health, fleet utilization, history, or mutations. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers, platform engineers, and Run:ai operators use this skill to survey read-only health across a filtered set of Run:ai workloads (by project, department, phase, node, node pool, age, image, or idle threshold), check failed workloads for node co-location patterns, or produce a five-section health snapshot of a single workload, then route confirmed failures to the matching diagnostic skill. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [Other [Run:ai MCP server read access; optional read-only Kubernetes access]] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Drill-down full health snapshot](references/drill-down-snapshot.md) <br>
- [Failed-workload node correlation](references/failed-node-correlation.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, API Calls] <br>
**Output Format:** [Markdown (workload overview tables or a five-section snapshot with a single HEALTHY/DEGRADED/FAILED/PENDING verdict)] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Read-only; overview output capped at a default limit of 50 rows; failed-node correlation capped at 50 workloads] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
30 evaluation tasks (22 positive, 8 negative), one attempt per task, each run in an isolated sandbox pod in a `k8s-sandbox` environment; baseline runs attempted the same tasks without the skill. <br>

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
| Overall | 84.4% (uplift unavailable) | 85.8% (uplift unavailable) |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 67.3% → 84.0% (+16.7 points) | 70.0% → 90.7% (+20.7 points) |
| Discoverability | 75.5% (uplift unavailable) | 92.5% (uplift unavailable) |
| Effectiveness | 44.3% → 64.9% (+20.6 points) | 48.8% → 79.0% (+30.2 points) |
| Efficiency | 97.4% (uplift unavailable) | 66.8% (uplift unavailable) |

## Skill Version(s): <br>
d524a853 (source: git SHA, committed 2026-10-25) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
