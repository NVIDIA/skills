## Description: <br>
Use as the top-level router for Omniverse Realtime Viewer USD app requests and focused viewer reference documents. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers building USD 3D viewer applications using NVIDIA Omniverse technologies, including local native viewers, cloud-streamed browser viewers, and hybrid desktop apps. <br>

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
- [Routing Reference](references/routing.md) <br>
- [Conventions Reference](references/conventions.md) <br>
- [Validation Reference](references/validation.md) <br>
- [USD Viewer App](references/usd-viewer-app/README.md) <br>
- [Streaming Viewer Recipe](references/streaming-viewer-recipe/README.md) <br>
- [OVUI Local Viewer Recipe](references/ovui-local-viewer-recipe/README.md) <br>
- [NVIDIA Skills Documentation](https://docs.nvidia.com/skills) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline code blocks and generated source files] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
7 evaluation tasks (7 positive) from the skill-evaluator-dataset-snapshot. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether the skill is safe to use. <br>
- Correctness: Checks whether the answer is correct. <br>
- Discoverability: Checks whether the right skill was loaded when needed. <br>
- Effectiveness: Checks whether the skill helped complete the task. <br>
- Efficiency: Checks whether the skill avoided wasted tool or skill usage. <br>

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
| Overall | 68% → 88% (+20 points) | 60% → 72% (+12 points) |
| Security | 93% → 86% (-7 points) | 50% → 7% (-43 points) |
| Correctness | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Discoverability | 42% → 92% (+50 points) | 45% → 79% (+34 points) |
| Effectiveness | 82% → 92% (+10 points) | 73% → 95% (+22 points) |
| Efficiency | 21% → 69% (+48 points) | 33% → 82% (+49 points) |

## Skill Version(s): <br>
0.2.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
