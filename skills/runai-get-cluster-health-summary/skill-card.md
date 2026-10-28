## Description: <br>
Point-in-time Run:ai cluster and node-health check that returns a HEALTHY, DEGRADED, or STUCK verdict per node pool with the dominant blocker (NotReady nodes, offline or failing GPU capacity, GPU counts, allocation vs utilization, the pending backlog, and a stuck scheduler vs a full pool), using read-only Run:ai MCP tools. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Run:ai cluster administrators and platform engineers use this skill to triage whether a cluster is healthy and what blocks each node pool right now, separating a stuck scheduler from a full pool and naming the next skill to hand off to, without making any changes to the cluster. <br>

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


## Skill Output: <br>
**Output Type(s):** [Analysis, API Calls] <br>
**Output Format:** [Markdown with a verdict line, a per-pool table, and short Scheduler, Nodes, Utilization, Failures, and Next sections] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Read-only; kept to one screen; point-in-time snapshot anchored to the health response asOf timestamp; pending backlog is sampled (default 5 oldest workloads) and the sample size is stated] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
22 evaluation tasks (15 positive, 7 negative), one attempt per task, each run with and without the skill in an isolated sandbox pod against replayed Run:ai MCP responses (one live-recorded healthy cluster and one synthetic cluster with one fault per node pool). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct? Final-answer correctness against the reference answer. <br>
- Discoverability: Was the right skill loaded when needed? <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight mean of tool-call productivity and token efficiency. <br>

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
| Overall | 89.7% (uplift unavailable) | 85.3% (uplift unavailable) |
| Security | 100.0% → 100.0% (±0.0 points) | 90.9% → 100.0% (+9.1 points) |
| Correctness | 79.1% → 88.2% (+9.1 points) | 76.4% → 91.8% (+15.4 points) |
| Discoverability | 88.0% (uplift unavailable) | 91.3% (uplift unavailable) |
| Effectiveness | 60.8% → 77.2% (+16.4 points) | 55.4% → 83.6% (+28.2 points) |
| Efficiency | 94.8% (uplift unavailable) | 59.5% (uplift unavailable) |

## Skill Version(s): <br>
779c097f (source: git SHA, committed 2026-10-28) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
