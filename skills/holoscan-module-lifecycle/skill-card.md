## Description: <br>
Develop external Holoscan Modules with the Holoscan CLI and use them in Holoscan applications: scaffold, test, Python bindings, DEB/WHEEL packaging, and clean-consumer proof. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to take a reusable external Holoscan Module from layout and API contract through scaffolding with the Holoscan CLI, implementation, testing, Python bindings, DEB/WHEEL packaging, and proof that a clean consumer application can install and use the module. <br>

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
- [Holoscan CLI contract](references/holoscan-cli-contract.md) <br>
- [Holoscan CLI configuration](references/holoscan-cli-configuration.md) <br>
- [Module development](references/module-development.md) <br>
- [Consumers and packaging](references/module-consumer-packaging.md) <br>
- [Holoscan Modules tutorials](https://github.com/nvidia-holoscan/holoscan-tutorials/tree/main/tutorials/holoscan-modules) <br>
- [Create a module tutorial](https://github.com/nvidia-holoscan/holoscan-tutorials/tree/main/tutorials/holoscan-modules/create-a-module) <br>
- [Use a module tutorial](https://github.com/nvidia-holoscan/holoscan-tutorials/tree/main/tutorials/holoscan-modules/use-a-module) <br>
- [Holoscan CLI quick start](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/README.md) <br>
- [Holoscan CLI command reference](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/CLI_REFERENCE.md) <br>
- [Holoscan CLI configuration guide](https://github.com/nvidia-holoscan/holoscan-cli/blob/main/CONFIGURATION.md) <br>
- [Holoscan applications repository](https://github.com/nvidia-holoscan/holoscan-applications) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Files, Shell commands, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks, plus generated module source, test, and packaging files] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Reports layout and identities, command preview and real results, honest test scope, consumer proof, artifact identity/contents/install results, and final worktree state; planning-only requests return a proposed workflow without executing commands.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
8 evaluation tasks (7 positive, 1 negative), 1 attempt per task, each run in an isolated k8s-sandbox pod, with and without the skill (baseline). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal. <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal. <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% `skill_efficiency` and 50% `token_efficiency`. <br>

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
| Overall | 95.1% — baseline ran, but no comparable score was available; uplift unavailable | 82.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 87.5% → 100.0% (+12.5 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 67.5% → 100.0% (+32.5 points) | 55.0% → 80.0% (+25.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 78.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 48.9% → 82.0% (+33.1 points) | 51.8% → 68.9% (+17.1 points) |
| Efficiency | 93.4% — baseline ran, but no comparable score was available; uplift unavailable | 86.5% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
376acb56 (source: git SHA, committed 2026-10-20) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
