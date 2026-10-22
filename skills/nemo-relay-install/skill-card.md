## Description: <br>
Use this skill when choosing or running NeMo Relay installation for an agent runtime, application language, framework integration, or plugin bundle, or checking a runtime's native Relay readiness. Do not use it for first-run instrumentation or export configuration. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to disambiguate the intended NeMo Relay install target (CLI/host integration, maintained framework or plugin integration, or an application-owned language binding), install the current stable package or executable from authoritative sources, and verify availability without extending into configuration, scopes, or a model-consuming trial. <br>

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
- [CLI Installation](references/cli-install.md) <br>
- [Language Packages](references/language-packages.md) <br>
- [Maintained Integrations](references/maintained-integrations.md) <br>
- [NeMo Relay Installation Guide](https://docs.nvidia.com/nemo/relay/getting-started/installation) <br>
- [NeMo Relay Support Matrix](https://docs.nvidia.com/nemo/relay/reference/support-matrix) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-sonnet-5-5`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
15 evaluation tasks (14 positive, 1 negative), each attempt run in its own isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct? Final-answer correctness against the reference answer. <br>
- Discoverability: Was the right skill loaded when needed, with decoys avoided? <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight mean of tool-call productivity and token efficiency. <br>

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
| Overall | 77.2% — uplift unavailable | 69.7% — uplift unavailable |
| Security | 70.0% → 76.7% (+6.7 points) | 73.3% → 76.7% (+3.4 points) |
| Correctness | 50.7% → 82.7% (+32.0 points) | 52.0% → 81.3% (+29.3 points) |
| Discoverability | 92.8% — uplift unavailable | 71.8% — uplift unavailable |
| Effectiveness | 35.9% → 59.0% (+23.1 points) | 27.1% → 53.1% (+26.0 points) |
| Efficiency | 75.0% — uplift unavailable | 65.7% — uplift unavailable |

## Skill Version(s): <br>
41cf25e2 (source: git SHA, committed 2026-10-22) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
