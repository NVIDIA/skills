## Description: <br>
Run the PAIDF Orchestration Image Attribute Augmentation DAG on Kubernetes — person-crop clothing augmentation, attribute search, and augmented dataset generation. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to run image attribute augmentation pipelines on Kubernetes, generating controlled clothing and attribute variations for person-crop datasets in physical AI use cases. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key, Cloud Credentials] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [airflow-direct-api.md](references/airflow-direct-api.md) <br>
- [outputs.md](references/outputs.md) <br>
- [payload-contract.md](references/payload-contract.md) <br>
- [setup-and-preflight.md](references/setup-and-preflight.md) <br>
- [troubleshooting.md](references/troubleshooting.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, API Calls] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
13 evaluation tasks (11 positive, 2 negative), each run in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed. <br>
- Effectiveness: Checks whether the user's goal was achieved and the expected workflow behavior was followed. <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 52% → 77% (+25 points) | 53% → 75% (+22 points) |
| Security | 85% → 100% (+15 points) | 81% → 96% (+15 points) |
| Correctness | 43% → 92% (+49 points) | 46% → 85% (+38 points) |
| Discoverability | 40% → 68% (+28 points) | 37% → 66% (+29 points) |
| Effectiveness | 46% → 75% (+30 points) | 48% → 66% (+18 points) |
| Efficiency | 45% → 48% (+4 points) | 53% → 62% (+9 points) |

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
