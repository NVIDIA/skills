## Description: <br>
Use when a user needs to get started with PAIDF Auto-Labeling, plan a scenario, run or debug a shipped cookbook, author prompts or cookbooks, migrate a pipeline, or configure a stage. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to plan, run, and debug PAIDF auto-labeling workflows that turn raw video and image datasets into annotation artifacts and training-ready outputs. <br>

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
- [PAIDF Auto-Labeling Skill References](references/README.md) <br>
- [Scenario Planning](references/scenario-planning.md) <br>
- [Cookbook Authoring](references/cookbook-authoring.md) <br>
- [Prompt Authoring](references/prompt-authoring.md) <br>
- [Pipeline Migration](references/pipeline-migration.md) <br>
- [Video Data Augmentation](references/video-data-augmentation.md) <br>
- [Event and Person Attribute Search](references/event-and-person-attribute-search.md) <br>
- [Event Verification Reasoning](references/event-verification-reasoning.md) <br>
- [Workflow Runner Debugging](references/workflow-runner-debugging.md) <br>
- [Workflow Stage Integration](references/workflow-stage-integration.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 3 internal evaluation tasks (3 positive). Each task attempt ran in its own isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Checks whether the user's goal was achieved and expected workflow behavior was followed. <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Verifies absence of unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `skill_execution`: Verifies whether the expected skill was found and executed. <br>
- `skill_efficiency`: Verifies routing quality, workspace-aware skill reads, and productive tool use. <br>
- `goal_accuracy`: Verifies whether the user's goal was achieved. <br>
- `behavior_check`: Verifies whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 52% → 95% (+43 points) | 53% → 96% (+43 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 33% → 100% (+67 points) | 40% → 100% (+60 points) |
| Discoverability | 50% → 99% (+49 points) | 46% → 90% (+44 points) |
| Effectiveness | 32% → 87% (+54 points) | 29% → 92% (+63 points) |
| Efficiency | 44% → 90% (+46 points) | 50% → 100% (+50 points) |

## Skill Version(s): <br>
1.1.0 (source: frontmatter, pyproject.toml, CHANGELOG) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
