## Description: <br>
Use when a user needs to get started with CDF Auto-Labeling, plan a scenario, run or debug a shipped cookbook, author prompts or cookbooks, migrate a pipeline, configure a stage, or pull the prebuilt stage containers from NGC (nvcr.io). <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to plan, configure, run, and debug CDF Auto-Labeling cookbooks that turn raw image and video datasets into annotation artifacts and training-ready outputs, including pulling the prebuilt stage containers from NGC. <br>

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
- [CDF Auto-Labeling Skill References](references/README.md) <br>
- [Scenario Planning](references/scenario-planning.md) <br>
- [Cookbook Authoring](references/cookbook-authoring.md) <br>
- [Prompt Authoring](references/prompt-authoring.md) <br>
- [Pipeline Migration](references/pipeline-migration.md) <br>
- [Video Data Augmentation](references/video-data-augmentation.md) <br>
- [Event and Person Attribute Search](references/event-and-person-attribute-search.md) <br>
- [Event Verification Reasoning](references/event-verification-reasoning.md) <br>
- [Workflow Runner Debugging](references/workflow-runner-debugging.md) <br>
- [Workflow Stage Integration](references/workflow-stage-integration.md) <br>
- [NGC Containers](references/ngc-containers.md) <br>
- [Stage References](references/stages/) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Files] <br>
**Output Format:** [Markdown with inline bash code blocks and YAML cookbook configurations] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 4 tasks (4 positive), 1 attempt per task, each in an isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use; checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed and decoys were avoided. <br>
- Effectiveness: Whether the skill helped complete the task (50% goal completion + 50% expected workflow adherence). <br>
- Efficiency: Whether wasted tool calls and token usage were avoided (50% tool-call productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 96.1% | 97.1% |
| Security | 100.0% → 100.0% (±0.0 points) | 62.5% → 100.0% (+37.5 points) |
| Correctness | 25.0% → 100.0% (+75.0 points) | 35.0% → 100.0% (+65.0 points) |
| Discoverability | 96.3% | 91.3% |
| Effectiveness | 29.6% → 95.6% (+66.0 points) | 25.0% → 96.3% (+71.3 points) |
| Efficiency | 88.4% | 97.7% |

## Skill Version(s): <br>
1.1.0 (source: frontmatter, pyproject.toml, changelog) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
