## Description: <br>
Top-level workflow skill for USD performance diagnosis and optimization that handles slow loading, high memory, low FPS, and broad scene-optimization requests. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to diagnose and optimize USD scene performance, addressing slow loading, high memory usage, low FPS, GPU crashes, and validation failures through structured profiling, structure assessment, and iterative optimization. <br>

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
- [Workflow Reference](references/workflow.md) <br>
- [Briefing the Skill](references/briefing-the-skill.md) <br>
- [Skill Map](references/skill-map.md) <br>
- [Setup USD Performance Tuning](references/setup-usd-performance-tuning/README.md) <br>
- [USD Structure Assessment](references/usd-structure-assessment/README.md) <br>
- [USD Validation Runner](references/usd-validation-runner/README.md) <br>
- [USD Optimize Run Operations](references/usd-optimize-run-operations/README.md) <br>
- [Optimization Report](references/optimization-report/README.md) <br>
- [Operations Registry](references/operations/README.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Files, Configuration instructions] <br>
**Output Format:** [Markdown reports, structured JSON, rendered HTML, optimized USD files] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Structured JSON conforms to optimization-report.schema.json; HTML rendered via render_preview.py; reports include before/after profile metrics] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
10 evaluation tasks (9 positive, 1 negative) from skill-evaluator-dataset-snapshot/1. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Verifies final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and followed expected workflow behavior. <br>
- Efficiency: Routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 55% → 91% (+37 points) | 55% → 91% (+36 points) |
| Security | 85% → 95% (+10 points) | 80% → 95% (+15 points) |
| Correctness | 40% → 94% (+54 points) | 44% → 86% (+42 points) |
| Discoverability | 51% → 99% (+48 points) | 51% → 88% (+38 points) |
| Effectiveness | 48% → 74% (+26 points) | 45% → 89% (+44 points) |
| Efficiency | 51% → 95% (+44 points) | 56% → 97% (+41 points) |

## Skill Version(s): <br>
0.4.1 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
