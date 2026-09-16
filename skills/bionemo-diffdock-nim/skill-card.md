## Description: <br>
Run DiffDock molecular docking via NVIDIA NIM to predict small-molecule binding poses against protein targets. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and computational biologists use this skill to predict protein-ligand binding poses via DiffDock on NVIDIA NIM, supporting drug discovery workflows including blind docking, pose ranking, and confidence scoring. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [DiffDock NIM API Reference](references/api.md) <br>
- [DiffDock Science Notes](references/science.md) <br>
- [DiffDock Parameter Guidance](references/parameters.md) <br>
- [DiffDock Validation](references/validation.md) <br>
- [DiffDock Examples](references/examples.md) <br>


## Skill Output: <br>
**Output Type(s):** [API Calls, Code, Files] <br>
**Output Format:** [JSON API responses with ranked SDF pose files] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Parallel arrays of ligand_positions (SDF strings) and position_confidence (numeric scores), rank-ordered by confidence] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
1 evaluation task (1 positive), 3 attempts per task, evaluated 2026-09-23 with evaluator version 1.5.6. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use, checking for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the skill produces correct answers against the reference answer. <br>
- Discoverability: Whether the right skill was selected and activated when needed, and decoys were avoided. <br>
- Effectiveness: Whether the skill helped complete the user's goal (50% goal_accuracy + 50% behavior_check). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage (50% tool productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity; routing scored under Discoverability. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 83.1% | 92.9% |
| Security | 50.0% → 100.0% (+50.0 points) | 50.0% → 100.0% (+50.0 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 100.0% | 90.0% |
| Effectiveness | 42.9% → 42.9% (±0.0 points) | 92.9% → 85.7% (-7.2 points) |
| Efficiency | 72.6% | 88.6% |

## Skill Version(s): <br>
0.1.0 (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
