## Description: <br>
Composes a new Kubernetes-only Airflow DAG from existing shared task groups in the PAIDF Orchestration repository, producing a K8s manifest, DAG builder, payload model, and workflow-local callables for custom synthetic data generation pipelines. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers building custom PAIDF Orchestration pipelines use this skill to compose new Kubernetes-only Airflow DAGs from shared task groups when no existing DAG covers the requested stage sequence. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [Cloud Credentials] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Component Skills Excerpt](references/component-skills-excerpt.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration files, Shell commands] <br>
**Output Format:** [Python files, YAML manifests, and Markdown summaries] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
7 evaluation tasks (7 positive) from a curated skill-evaluator dataset snapshot. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal (goal completion and expected workflow adherence). <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 39% → 85% (+46 points) | 38% → 71% (+33 points) |
| Security | 71% → 100% (+29 points) | 71% → 50% (-21 points) |
| Correctness | 37% → 89% (+51 points) | 34% → 89% (+54 points) |
| Discoverability | 50% → 98% (+48 points) | 37% → 74% (+38 points) |
| Effectiveness | 12% → 56% (+43 points) | 11% → 53% (+42 points) |
| Efficiency | 24% → 84% (+59 points) | 39% → 91% (+52 points) |

## Skill Version(s): <br>
1.1.0 (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
