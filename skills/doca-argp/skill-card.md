## Description: <br>
Guides hands-on DOCA Arg Parser CLI work on a shipped DOCA sample or new DOCA-using app, covering adding, removing, or renaming flags, wiring the doca_argp init/register/start/destroy lifecycle in order, picking a parameter type from the six-value public enum, preserving the standard --device / --representor / --json / --sdk-log-level surface, and debugging DOCA_ERROR codes returned from doca_argp calls. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
External developers and engineers building or modifying DOCA-using applications (C/C++ or FFI bindings) use this skill to register, configure, and debug DOCA Arg Parser flags and JSON config input while keeping the standard DOCA sample CLI surface. <br>

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
- [DOCA Arg Parser Documentation](https://docs.nvidia.com/doca/sdk/DOCA-Arg-Parser/index.html) <br>
- [CAPABILITIES.md](CAPABILITIES.md) <br>
- [TASKS.md](TASKS.md) <br>
- [DOCA Samples](https://github.com/NVIDIA-DOCA/doca-samples) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Code, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), 1 attempt per task, each run in an isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether the skill is safe to use (unsafe operations, secret leakage, unauthorized access). <br>
- Correctness: Checks whether the final answer is correct against the reference answer. <br>
- Discoverability: Checks whether the right skill was loaded when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal and expected workflow. <br>
- Efficiency: Checks whether wasted tool calls and token usage were avoided. <br>

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
| Overall | 98.2% (no comparable baseline) | 98.4% (no comparable baseline) |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 80.0% → 100.0% (+20.0 points) | 85.0% → 100.0% (+15.0 points) |
| Discoverability | 100.0% (no comparable baseline) | 95.0% (no comparable baseline) |
| Effectiveness | 70.0% → 100.0% (+30.0 points) | 60.0% → 100.0% (+40.0 points) |
| Efficiency | 90.8% (no comparable baseline) | 96.8% (no comparable baseline) |

## Skill Version(s): <br>
7e143d8 (source: git SHA, committed 2026-10-12) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
