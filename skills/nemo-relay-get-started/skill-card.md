## Description: <br>
Use this skill when first-time NeMo Relay users want help selecting and following an appropriate quick-start guide or verifying their first working integration. Do not use it for production rollout or broad integration development. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
First-time NeMo Relay developers and engineers use this skill to select and follow the maintained quick-start guide that fits their existing runtime, framework, plugin, or application, and to verify their first working integration with the guide's own success criteria. <br>

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
- [CLI Trial Routing](references/cli-try-now.md) <br>
- [Agent Runtime Trial Routing](references/agent-runtimes-try-now.md) <br>
- [Framework Trial Routing](references/built-in-integrations-try-now.md) <br>
- [Switchyard Trial Routing](references/switchyard-try-now.md) <br>
- [Language Trial Routing](references/manual-language-try-now.md) <br>
- [Guide-Based First-Value Checks](references/first-value.md) <br>
- [NeMo Relay Quick-Start Index](https://docs.nvidia.com/nemo/relay/getting-started/quick-start) <br>
- [NeMo Relay Stable Plugin Overview](https://docs.nvidia.com/nemo/relay/latest/configure-plugins/about) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline shell and configuration code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-sonnet-5-5`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
22 evaluation tasks (21 positive, 1 negative), one attempt per task, each run in its own isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use, covering unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded and activated when needed and decoys were avoided. <br>
- Effectiveness: Whether the skill helped complete the user's goal and followed the expected workflow. <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 81.0% (uplift unavailable) | 71.3% (uplift unavailable) |
| Security | 86.4% → 86.4% (±0.0 points) | 79.6% → 70.5% (-9.1 points) |
| Correctness | 33.6% → 84.6% (+51.0 points) | 56.4% → 85.5% (+29.1 points) |
| Discoverability | 92.7% (uplift unavailable) | 83.8% (uplift unavailable) |
| Effectiveness | 34.4% → 62.9% (+28.5 points) | 38.7% → 55.5% (+16.8 points) |
| Efficiency | 78.5% (uplift unavailable) | 61.0% (uplift unavailable) |

## Skill Version(s): <br>
41cf25e2 (source: git SHA, committed 2026-10-22) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
