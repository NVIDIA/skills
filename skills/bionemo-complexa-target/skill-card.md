## Description: <br>
Add, edit, inspect, or validate Proteina-Complexa target definitions for protein binders, ligand binders, and AME motif scaffolding via direct YAML edits or complexa target/validate target commands. <br>

This skill is for research and development only. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and computational biologists use this skill to add, edit, inspect, or validate Proteina-Complexa design target definitions — protein binders, ligand binders, and AME motif scaffolds — editing the appropriate targets dict YAML, preserving existing entries, and reporting the saved entry, its consumer pipeline, the downstream task name, and what validation ran. <br>

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
- [Target schema reference](references/target_schema.md) <br>
- [Proteina-Complexa Hardware Reference](references/hardware.md) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Files, Shell commands] <br>
**Output Format:** [Markdown with inline YAML/bash code blocks, plus a saved YAML target definition entry] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Writes the edited target entry to the requested output location (default ./target_<name>/ or target_definition.yaml); the CLI path also produces an automatic .yaml.bak backup. Coordinate/scientific validation stays pending when no target structure is present.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), one attempt per task, each run in an isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is the skill safe to use? Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the final answer correct against the reference answer? <br>
- Discoverability: Was the expected skill selected and activated when needed while decoys were avoided? <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did the run avoid wasted tool calls and token usage? <br>

Underlying evaluation signals used in this run: <br>
- `security`: Flags unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 94.7% — baseline ran, but no comparable score was available; uplift unavailable | 82.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 25.0% → 50.0% (+25.0 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 95.0% — baseline ran, but no comparable score was available; uplift unavailable | 86.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 97.5% → 100.0% (+2.5 points) | 97.5% → 100.0% (+2.5 points) |
| Efficiency | 78.4% — baseline ran, but no comparable score was available; uplift unavailable | 74.0% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
1.1.0 (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
