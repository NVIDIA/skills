## Description: <br>
Use as the top-level router for Omniverse Realtime Viewer USD app requests and focused viewer reference documents. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers building realtime USD viewer applications using NVIDIA Omniverse RTX rendering, streaming, and interaction capabilities. <br>

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
- [Routing](references/routing.md) <br>
- [Conventions](references/conventions.md) <br>
- [USD Viewer App](references/usd-viewer-app/README.md) <br>
- [Streaming vs Local](references/streaming-vs-local/README.md) <br>
- [Streaming Viewer Recipe](references/streaming-viewer-recipe/README.md) <br>
- [OVUI Local Viewer Recipe](references/ovui-local-viewer-recipe/README.md) <br>
- [Electron SHM Viewer](references/electron-shm-viewer/README.md) <br>
- [OVStage Runtime](references/ovstage-runtime/README.md) <br>
- [Cloud Deployment](references/cloud-deployment/README.md) <br>
- [Validation](references/validation.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions, Shell commands] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 7 evaluation tasks (7 positive) using the skill-evaluator three-tier framework (Tier 1: static validation, Tier 3: live agent evaluation). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Verifies final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Measures goal completion and expected workflow adherence. <br>
- Efficiency: Evaluates routing quality, workspace-aware skill reads, and productive tool use. <br>

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
| Overall | Not available | 57% → 75% (+18 points) |
| Security | Not available | 36% → 14% (-21 points) |
| Correctness | Not available | 100% → 100% (±0 points) |
| Discoverability | Not available | 44% → 77% (+33 points) |
| Effectiveness | Not available | 73% → 98% (+26 points) |
| Efficiency | Not available | 31% → 84% (+52 points) |

## Skill Version(s): <br>
0.2.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
