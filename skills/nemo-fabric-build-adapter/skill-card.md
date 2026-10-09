## Description: <br>
Build, migrate, review, and maintain third-party NVIDIA NeMo Fabric adapters against the public adapter contract. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers creating, migrating, reviewing, or maintaining third-party adapters that integrate external agent harnesses or custom-agent runtimes with NVIDIA NeMo Fabric through the published southbound adapter contract. <br>

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
- [Adapter Contract Documentation](https://github.com/NVIDIA/NeMo-Fabric/tree/main/docs/adapter-contract) <br>
- [Adapter Contract JSON Schemas](https://github.com/NVIDIA/NeMo-Fabric/tree/main/schemas/adapter-contract) <br>
- [NeMo Agent Toolkit Shared Adapter](https://github.com/NVIDIA/NeMo-Fabric/tree/main/external/nat) <br>
- [LangGraph Custom Agent Example](https://github.com/NVIDIA/NeMo-Fabric/tree/main/examples/langgraph_custom_agent) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
7 evaluation tasks (5 positive, 2 negative) executed in isolated sandbox pods with 1 attempt per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal. <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal. <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% `skill_efficiency` plus 50% `token_efficiency`. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | Not available | 78.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 57.1% → 85.7% (+28.6 points) |
| Correctness | Not available | 71.4% → 82.9% (+11.5 points) |
| Discoverability | Not available | 82.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 52.1% → 70.2% (+18.1 points) |
| Efficiency | Not available | 70.6% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
ac263cc (source: git SHA, committed 2026-10-08) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
