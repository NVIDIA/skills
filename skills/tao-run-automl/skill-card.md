## Description: <br>
Run container-backed AutoML / hyperparameter optimization (HPO) for NVIDIA TAO networks using AutoMLRunner. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers running automated hyperparameter optimization for NVIDIA TAO deep-learning models across platforms (Brev, SLURM, Kubernetes, Docker). <br>

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
- [AutoML Preflight Concepts](references/automl-preflight-concepts.md) <br>
- [AutoML Intent Algorithms](references/automl-intent-algorithms.md) <br>
- [AutoML Runner Configuration](references/automl-runner-configuration.md) <br>
- [AutoML Advanced Monitoring](references/automl-advanced-monitoring.md) <br>
- [AutoML Compression Literature](references/automl-compression-literature.md) <br>
- [AutoML Examples](references/automl-examples.md) <br>
- [Detailed Guide](references/detailed-guide.md) <br>
- [Skill Info](references/skill_info.yaml) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Code, Analysis] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
2 evaluation tasks (2 positive), evaluated against internal skill-evaluator-dataset-snapshot/1. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Verifies final-answer correctness against reference answers. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Detects unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Routing quality and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 64% → 97% (+33 points) | 57% → 78% (+21 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 70% → 100% (+30 points) | 100% → 100% (±0 points) |
| Discoverability | 50% → 100% (+50 points) | 0% → 47% (+47 points) |
| Effectiveness | 58% → 100% (+42 points) | 85% → 92% (+8 points) |
| Efficiency | 41% → 83% (+43 points) | 0% → 50% (+50 points) |

## Skill Version(s): <br>
0.1.1 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
