## Description: <br>
Customize NVIDIA Nemotron Voice Agent's Generic Pipecat example for healthcare appointment, five-field patient intake, or custom tool-calling workflows without a separate backend. <br>

This skill is for demonstration purposes and not for production usage. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers customizing NVIDIA Nemotron Voice Agent for healthcare voice workflows such as appointment scheduling, patient intake, or custom tool-calling scenarios. <br>

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
- [NVIDIA Nemotron Voice Agent Blueprint](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent) <br>
- [Deployment Modes](references/deployment-modes.md) <br>
- [Example Design Choices](references/example-design-choices.md) <br>
- [User Experience Flowchart](references/user-experience-flowchart.md) <br>
- [Custom Workflow Guide](references/custom/guide.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Code] <br>
**Output Format:** [Markdown with inline bash and Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), 3 attempts per task, each in an isolated sandbox pod. Dataset digest: sha256:d8ae4021cef430b548b0c29cac10d8517b49a7c5910f714f06a474939295ff30. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Equal-weight mean of goal completion (goal_accuracy) and expected workflow adherence (behavior_check). <br>
- Efficiency: 50% tool-call productivity and 50% token efficiency measuring actual uncached prompt plus completion usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Detects unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (routing scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 90.3% | 88.3% |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 14.3% → 85.0% (+70.7 points) | 33.3% → 95.0% (+61.7 points) |
| Discoverability | 100.0% | 61.7% |
| Effectiveness | 50.0% → 82.9% (+32.9 points) | 34.3% → 88.3% (+54.0 points) |
| Efficiency | 83.5% | 96.7% |

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
