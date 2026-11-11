## Description: <br>
Composes a new Kubernetes Airflow DAG (with an optional external-services-only OSMO variant) from the existing shared and workflow-local CDF Orchestration task groups and stages when a user describes a custom ordered pipeline that no existing DAG in airflow/dags/workflows/ covers, and checks generated or existing DAG model, container, and prompt choices against an external spec document. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to author, verify, and (with explicit confirmation) deploy, trigger, and monitor custom Cosmos Data Factory (CDF) Orchestration Airflow DAGs that chain existing task groups and stages, such as augmentation-only, auto-labeling-only, or detection-and-captioning pipelines, for synthetic data generation. <br>

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
- [Component Skills — Minimal Excerpt](references/component-skills-excerpt.md) <br>
- [GPU Footprint Derivation](references/gpu-footprint.md) <br>
- [Trigger Examples](references/trigger-examples.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Files, Shell commands, API Calls] <br>
**Output Format:** [Python and YAML files plus a Markdown summary with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Deployment (make build/install) and DAG run triggering require explicit user confirmation] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
9 evaluation tasks (7 positive, 2 negative), 1 attempt per task, each run in an isolated sandbox pod in a k8s-sandbox environment. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal. <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal. <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight mean of `skill_efficiency` and `token_efficiency`. <br>

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
| Overall | 87.2% — baseline ran, but no comparable score was available; uplift unavailable | 76.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 83.3% → 100.0% (+16.7 points) | 66.7% → 72.2% (+5.5 points) |
| Correctness | 33.3% → 95.6% (+62.3 points) | 31.1% → 73.3% (+42.2 points) |
| Discoverability | 95.7% — baseline ran, but no comparable score was available; uplift unavailable | 82.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 24.5% → 66.6% (+42.1 points) | 15.8% → 67.3% (+51.5 points) |
| Efficiency | 78.3% — baseline ran, but no comparable score was available; uplift unavailable | 87.7% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
1.0.1 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
