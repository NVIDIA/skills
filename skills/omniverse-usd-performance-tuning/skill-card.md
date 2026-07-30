## Description: <br>
Top-level workflow skill for USD performance diagnosis and optimization. Handles slow loading, high memory, low FPS, and broad scene-optimization requests; delegates auth/runtime setup to Phase 0 owners. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers working with USD scenes who need to diagnose and resolve performance problems such as slow loading, high memory usage, low FPS, GPU device-lost events, and validation failures. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [OAuth Token] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Workflow Reference](references/workflow.md) <br>
- [Skill Map](references/skill-map.md) <br>
- [Operations Registry](references/operations/README.md) <br>
- [USD Structure Assessment](references/usd-structure-assessment/README.md) <br>
- [Optimization Report](references/optimization-report/README.md) <br>
- [USD Validation Runner](references/usd-validation-runner/README.md) <br>
- [Usd Optimize Run Operations](references/usd-optimize-run-operations/README.md) <br>
- [Setup USD Performance Tuning](references/setup-usd-performance-tuning/README.md) <br>


## Skill Output: <br>
**Output Type(s):** [Files, Analysis, Shell commands] <br>
**Output Format:** [Optimized USD stage, JSON report, Markdown summary, rendered HTML report] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Structured JSON conforms to optimization-report schema; HTML rendered via template] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
9 evaluation tasks (8 positive, 1 negative), each running in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and followed expected workflow behavior. <br>
- Efficiency: Whether the skill avoided wasted tool or skill usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Codex (Baseline → Skill Uplift) |
|---|---:|
| Overall | 46% → 84% (+38 points) |
| Security | 89% → 100% (+11 points) |
| Correctness | 20% → 76% (+56 points) |
| Discoverability | 47% → 91% (+44 points) |
| Effectiveness | 37% → 63% (+26 points) |
| Efficiency | 35% → 91% (+56 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
