## Description: <br>
Top-level workflow skill for USD performance diagnosis and optimization. Handles slow loading, high memory, low FPS, and broad scene-optimization requests; delegates auth/runtime setup to Phase 0 owners. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to diagnose and optimize USD scene performance, addressing slow loading, high memory consumption, low FPS, and GPU resource issues in NVIDIA Omniverse workflows. <br>

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
- [Skill Map](references/skill-map.md) <br>
- [Briefing the Skill](references/briefing-the-skill.md) <br>
- [Operations Registry](references/operations/README.md) <br>
- [USD Structure Assessment](references/usd-structure-assessment/README.md) <br>
- [USD Validation Runner](references/usd-validation-runner/README.md) <br>
- [Optimization Report](references/optimization-report/README.md) <br>
- [Upstream USD Optimize](references/upstreams/usd-optimize.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions, Files] <br>
**Output Format:** [Markdown with inline code blocks, structured JSON reports, and rendered HTML] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Produces optimization-report JSON, Markdown summary, and HTML preview via report templates] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
10 evaluation tasks (9 positive, 1 negative) from a curated dataset snapshot. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Checks goal completion (50%) and expected workflow adherence (50%). <br>
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
| Overall | 52% → 89% (+37 points) | 56% → 85% (+29 points) |
| Security | 95% → 100% (+5 points) | 90% → 95% (+5 points) |
| Correctness | 36% → 92% (+56 points) | 54% → 82% (+28 points) |
| Discoverability | 46% → 94% (+48 points) | 49% → 84% (+34 points) |
| Effectiveness | 42% → 71% (+29 points) | 41% → 66% (+25 points) |
| Efficiency | 43% → 90% (+47 points) | 47% → 98% (+51 points) |

## Skill Version(s): <br>
0.4.1 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
