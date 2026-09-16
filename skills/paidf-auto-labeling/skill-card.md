## Description: <br>
Use when a user needs to get started with PAIDF Auto-Labeling, plan a scenario, run or debug a shipped cookbook, author prompts or cookbooks, migrate a pipeline, or configure a stage. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to plan, configure, run, and debug PAIDF Auto-Labeling cookbooks that turn raw image and video datasets into annotation artifacts and training-ready outputs. <br>

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
Evaluated against 3 tasks (3 positive) in isolated k8s-sandbox pods, 3 attempts per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use — checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed and decoys were avoided. <br>
- Effectiveness: Whether the skill helped complete the user's goal (50% goal completion + 50% expected workflow adherence). <br>
- Efficiency: Whether wasted tool calls and token usage were avoided (50% tool-call productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 94.7% | 93.7% |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 14.3% → 100.0% (+85.7 points) | 36.0% → 100.0% (+64.0 points) |
| Discoverability | 95.0% | 88.3% |
| Effectiveness | 18.6% → 90.8% (+72.2 points) | 23.5% → 87.5% (+64.0 points) |
| Efficiency | 87.8% | 92.8% |

## Skill Version(s): <br>
1.1.0 (source: frontmatter, pyproject.toml, changelog) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
