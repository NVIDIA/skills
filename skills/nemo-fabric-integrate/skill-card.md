## Description: <br>
Use this skill when integrating NVIDIA NeMo Fabric into a consumer application, service, evaluation harness, or platform through the typed Python SDK — translating the consumer's own application, job, or deployment config into an in-memory FabricConfig, choosing the single-invocation convenience API or an explicitly started runtime, validating with plan and doctor, and consuming normalized results, artifacts, and telemetry. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers integrating NVIDIA NeMo Fabric into consumer applications, services, evaluation harnesses, or platforms through the typed Python SDK. <br>

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
- [Config Mapping Reference](references/config-mapping.md) <br>
- [Results and Errors Reference](references/results-and-errors.md) <br>
- [SDK API Inventory](references/sdk-api-inventory.md) <br>
- [Python SDK Guide](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/sdk/python.mdx) <br>
- [NeMo Fabric Overview](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/about-nemo-fabric/overview.mdx) <br>
- [Installation Guide](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/getting-started/install.mdx) <br>
- [Code Review Agent Example](https://github.com/NVIDIA/NeMo-Fabric/tree/main/examples/code_review_agent) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions] <br>
**Output Format:** [Markdown with inline Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
5 evaluation tasks (3 positive, 2 negative) from skill-evaluator-dataset-snapshot/1. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helps complete the user's goal and follows expected workflow behavior. <br>
- Efficiency: Whether the skill avoids wasted tool or skill usage through quality routing and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `skill_execution`: Verifies the expected skill was found and executed. <br>
- `goal_accuracy`: Checks whether the user's goal was achieved. <br>
- `behavior_check`: Checks whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 57% → 87% (+30 points) | 56% → 79% (+23 points) |
| Security | 90% → 100% (+10 points) | 40% → 40% (±0 points) |
| Correctness | 32% → 84% (+52 points) | 60% → 88% (+28 points) |
| Discoverability | 66% → 91% (+25 points) | 61% → 84% (+22 points) |
| Effectiveness | 44% → 73% (+29 points) | 56% → 87% (+31 points) |
| Efficiency | 51% → 86% (+34 points) | 62% → 97% (+35 points) |

## Skill Version(s): <br>
134c829 (source: git SHA, committed 2026-08-14) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
