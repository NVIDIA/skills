## Description: <br>
Build, migrate, review, and maintain third-party NVIDIA NeMo Fabric adapters against the public adapter contract. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers building third-party adapters for the NVIDIA NeMo Fabric southbound contract, including creating adapter and target descriptors, mapping normalized configuration, implementing lifecycle methods, and verifying adapter conformance. <br>

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
- [NVIDIA NeMo Fabric Adapter Contract](https://github.com/NVIDIA/NeMo-Fabric/tree/main/docs/adapter-contract) <br>
- [Adapter-Contract JSON Schemas](https://github.com/NVIDIA/NeMo-Fabric/tree/main/schemas/adapter-contract) <br>
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
4 evaluation tasks (2 positive, 2 negative) from skill-evaluator-dataset-snapshot. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use: checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the skill produces correct answers against the reference answer. <br>
- Discoverability: Whether the right skill was found and activated when needed. <br>
- Effectiveness: Whether the skill helps complete the user's goal and expected workflow (goal completion and behavior adherence). <br>
- Efficiency: Whether the skill avoids wasted tool or skill usage through good routing and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 63% → 87% (+24 points) | 60% → 88% (+28 points) |
| Security | 100% → 75% (-25 points) | 50% → 75% (+25 points) |
| Correctness | 30% → 100% (+70 points) | 55% → 100% (+45 points) |
| Discoverability | 74% → 94% (+19 points) | 83% → 92% (+9 points) |
| Effectiveness | 46% → 80% (+34 points) | 48% → 83% (+36 points) |
| Efficiency | 66% → 88% (+23 points) | 66% → 89% (+23 points) |

## Skill Version(s): <br>
134c829 (source: git SHA, committed 2026-08-14) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
