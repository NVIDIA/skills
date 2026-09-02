## Description: <br>
Use when a user describes a custom PAIDF Orchestration pipeline — a specific ordered combination of stages such as augmentation only, auto-labeling only, detection+captioning only, or image attribute augmentation without full auto-labeling — that no existing DAG covers, and asks for a new Kubernetes DAG; also use to check that a generated or existing DAG’s model/container/prompt choices match an external spec document. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to compose custom Kubernetes-only Airflow DAGs for Physical AI Data Factory (PAIDF) orchestration pipelines, assembling specific ordered combinations of shared task groups for synthetic data generation workflows that no existing checked-in DAG covers. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [Other [Kubernetes cluster credential file]] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Component Skills Minimal Excerpt](references/component-skills-excerpt.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration files, Shell commands] <br>
**Output Format:** [Python files, YAML manifests, and Markdown summaries] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
7 evaluation tasks (7 positive), each run in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks whether the final answer is correct against the reference answer. <br>
- Discoverability: Checks whether the right skill was found and executed when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user’s goal (goal completion 50% + expected workflow adherence 50%). <br>
- Efficiency: Checks whether the skill avoided wasted tool or skill usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `goal_accuracy`: Whether the user’s goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 41% → 89% (+48 points) | 30% → 82% (+52 points) |
| Security | 71% → 100% (+29 points) | 43% → 93% (+50 points) |
| Correctness | 37% → 97% (+60 points) | 31% → 91% (+60 points) |
| Discoverability | 48% → 99% (+51 points) | 38% → 79% (+41 points) |
| Effectiveness | 20% → 65% (+45 points) | 9% → 65% (+56 points) |
| Efficiency | 29% → 85% (+57 points) | 30% → 80% (+50 points) |

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
