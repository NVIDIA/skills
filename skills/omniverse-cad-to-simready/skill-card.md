## Description: <br>
Coordinate the end-to-end CAD/source-asset to SimReady workflow, including conversion, material/physics assignment, SimReady conformance, validation, and optional package creation. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers converting CAD or source assets into simulation-ready USD packages with automated property assignment, conformance validation, and optional packaging. <br>

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
- [Commands Reference](references/commands.md) <br>
- [Preflight Setup](references/preflight/README.md) <br>
- [Convert to USD](references/convert-to-usd/README.md) <br>
- [Content Agents](references/content-agents/README.md) <br>
- [SimReady Conform Profile](references/simready-conform-profile/README.md) <br>
- [SimReady Validate](references/simready-validate/README.md) <br>
- [Asset Validator](references/omni-asset-validate/README.md) <br>
- [OVRTX Render Service](references/ovrtx-render-service/README.md) <br>
- [Troubleshooting](references/troubleshooting.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Files] <br>
**Output Format:** [Markdown with inline JSON structured artifacts] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Consolidated workflow report with stage statuses, validation findings, and output USD paths] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 8 tasks (7 positive, 1 negative) in isolated k8s-sandbox pods with 1 attempt per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill is found and executed when needed. <br>
- Effectiveness: Whether the skill helps the agent complete the user's goal and expected workflow. <br>
- Efficiency: Whether the skill avoids wasted tool or skill usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Codex (Baseline → Skill Uplift) |
|---|---:|
| Overall | 48% → 82% (+33 points) |
| Security | 50% → 81% (+31 points) |
| Correctness | 57% → 88% (+30 points) |
| Discoverability | 48% → 81% (+34 points) |
| Effectiveness | 37% → 75% (+38 points) |
| Efficiency | 49% → 83% (+34 points) |

## Testing Completed: <br>
**[x] Agent Red-Teaming** <br>
**[ ] Network Security** <br>
**[ ] Product Security** <br>

## Skill Version(s): <br>
0.2.0 (source: frontmatter, changelog) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
