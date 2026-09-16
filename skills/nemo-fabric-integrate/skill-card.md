## Description: <br>
Use this skill when integrating NVIDIA NeMo Fabric into a consumer application, service, evaluation harness, or platform through the typed Python SDK — translating the consumer’s own application, job, or deployment config into an in-memory FabricConfig, choosing the single-invocation convenience API or an explicitly started runtime, validating with plan and doctor, and consuming normalized results, artifacts, and telemetry. <br>

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
- [config-mapping.md](references/config-mapping.md) <br>
- [results-and-errors.md](references/results-and-errors.md) <br>
- [sdk-api-inventory.md](references/sdk-api-inventory.md) <br>
- [Python SDK Guide](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/sdk/python.mdx) <br>
- [NeMo Fabric Overview](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/about-nemo-fabric/overview.mdx) <br>
- [Installation Guide](https://github.com/NVIDIA/NeMo-Fabric/blob/main/docs/getting-started/install.mdx) <br>
- [code_review_agent Example](https://github.com/NVIDIA/NeMo-Fabric/tree/main/examples/code_review_agent) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions] <br>
**Output Format:** [Markdown with inline Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
6 evaluation tasks (4 positive, 2 negative), each run in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user’s goal and expected workflow (equal-weight mean of goal completion and behavior check). <br>
- Efficiency: Routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `goal_accuracy`: Whether the user’s goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 51% → 79% (+28 points) | 54% → 80% (+26 points) |
| Security | 100% → 67% (-33 points) | 83% → 58% (-25 points) |
| Correctness | 10% → 87% (+77 points) | 33% → 87% (+53 points) |
| Discoverability | 66% → 93% (+27 points) | 59% → 86% (+27 points) |
| Effectiveness | 22% → 70% (+48 points) | 33% → 73% (+40 points) |
| Efficiency | 57% → 78% (+21 points) | 60% → 96% (+36 points) |

## Skill Version(s): <br>
ae47327 (source: git SHA, committed 2026-09-02) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
