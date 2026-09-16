## Description: <br>
Coordinate the end-to-end CAD/source-asset to SimReady workflow. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers converting CAD and source assets into simulation-ready (SimReady) USD assets through an automated multi-stage workflow covering conversion, material and physics assignment, SimReady conformance, validation, rendering, and optional packaging. <br>

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
- [Workflow Reference](references/workflow.md) <br>
- [Command Patterns](references/commands.md) <br>
- [Preflight Setup](references/preflight/README.md) <br>
- [Convert to USD](references/convert-to-usd/README.md) <br>
- [Content Agents](references/content-agents/README.md) <br>
- [Deploy Content Agents](references/deploy-content-agents/README.md) <br>
- [SimReady Conform Profile](references/simready-conform-profile/README.md) <br>
- [SimReady Validate](references/simready-validate/README.md) <br>
- [Validate USD Minimum](references/validate-usd-minimum/README.md) <br>
- [Asset Validate](references/omni-asset-validate/README.md) <br>
- [Geometry Validate](references/omni-asset-validate-geometry/README.md) <br>
- [Physics Validate](references/omni-asset-validate-physics/README.md) <br>
- [OVRTX Render Service](references/ovrtx-render-service/README.md) <br>
- [Assemble Package Source](references/assemble-package-source/README.md) <br>
- [Troubleshooting](references/troubleshooting.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Files, Shell commands] <br>
**Output Format:** [Markdown with inline JSON artifacts] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
8 evaluation tasks (7 positive, 1 negative) across a three-tier evaluation (static validation, semantic deduplication, live agent evaluation) in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Verifies final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Equal-weight mean of goal completion and expected workflow adherence. <br>
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
| Overall | 54% → 91% (+38 points) | Not available |
| Security | 75% → 100% (+25 points) | Not available |
| Correctness | 57% → 100% (+43 points) | Not available |
| Discoverability | 48% → 97% (+49 points) | Not available |
| Effectiveness | 47% → 72% (+24 points) | Not available |
| Efficiency | 42% → 89% (+47 points) | Not available |

## Skill Version(s): <br>
0.2.0 (source: frontmatter, changelog) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
