## Description: <br>
Set up or troubleshoot the Auto Ontology runtime, covering Helm (the official install), Docker Compose, developer setup, and MCP connection to an existing deployment. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and platform engineers use this skill to install Auto Ontology with the published Helm chart or Docker Compose, run developer workflows, connect a source database, verify readiness, and connect an MCP client to a deployment that is already running. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key, OAuth Token, Other [Basic auth]] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Auto Ontology setup troubleshooting](references/troubleshooting.md) <br>
- [runtime-contract.yaml](assets/runtime-contract.yaml) <br>
- [Table-level scope guarantee (NVIDIA/auto-ontology#255)](https://github.com/NVIDIA/auto-ontology/issues/255) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, API Calls] <br>
**Output Format:** [Markdown with inline bash and YAML code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Secrets are never requested in the conversation or placed in command arguments; users write them into a values file or .env themselves.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
10 positive evaluation tasks, one attempt per task, each run in an isolated k8s-sandbox pod and compared against a no-skill baseline. <br>

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
| Overall | 95.6% — baseline ran, but no comparable score was available; uplift unavailable | 89.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 90.0% → 100.0% (+10.0 points) | 60.0% → 85.0% (+25.0 points) |
| Correctness | 36.0% → 100.0% (+64.0 points) | 68.0% → 100.0% (+32.0 points) |
| Discoverability | 94.0% — baseline ran, but no comparable score was available; uplift unavailable | 87.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 42.8% → 95.5% (+52.7 points) | 54.5% → 88.5% (+34.0 points) |
| Efficiency | 88.3% — baseline ran, but no comparable score was available; uplift unavailable | 87.2% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
0.3.1 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
