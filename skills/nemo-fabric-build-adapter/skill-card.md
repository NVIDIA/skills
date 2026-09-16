## Description: <br>
Build, migrate, review, and maintain third-party NVIDIA NeMo Fabric adapters against the public adapter contract. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers creating, migrating, reviewing, or maintaining third-party adapters that integrate external agent runtimes with NVIDIA NeMo Fabric through the published southbound adapter contract. <br>

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
6 evaluation tasks (4 positive, 2 negative) executed in isolated sandbox pods with 1 attempt per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the skill produces correct answers against the reference answer. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helps the agent complete the user's goal and expected workflow (equal-weight mean of goal completion and workflow adherence). <br>
- Efficiency: Whether the skill avoids wasted tool or skill usage through quality routing and productive tool use. <br>

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
| Overall | Not available | 52% → 79% (+27 points) |
| Security | Not available | 83% → 50% (-33 points) |
| Correctness | Not available | 30% → 87% (+57 points) |
| Discoverability | Not available | 58% → 86% (+28 points) |
| Effectiveness | Not available | 27% → 81% (+54 points) |
| Efficiency | Not available | 61% → 90% (+30 points) |

## Skill Version(s): <br>
ae47327 (source: git SHA, committed 2026-09-02) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
