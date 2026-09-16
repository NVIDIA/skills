## Description: <br>
Internal NVFLARE conversion references and templates. Use only when another NVFLARE skill directs you to a shared workflow, policy, or asset. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers building federated learning conversion workflows with NVFLARE agent skills use this shared reference skill to load canonical conversion policies, templates, and validation guidance. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [No] <br>
**Credential Type(s):** [None] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [conversion-common.md](references/conversion-common.md) <br>
- [conversion-workflow.md](references/conversion-workflow.md) <br>
- [dependency-install.md](references/dependency-install.md) <br>
- [metrics-and-artifact-reporting.md](references/metrics-and-artifact-reporting.md) <br>
- [pytorch-family-recipe-construction.md](references/pytorch-family-recipe-construction.md) <br>
- [pytorch-family-recipe-selection.md](references/pytorch-family-recipe-selection.md) <br>
- [pytorch-model-exchange.md](references/pytorch-model-exchange.md) <br>
- [runtime-output-guidance.md](references/runtime-output-guidance.md) <br>
- [site-data-and-paths.md](references/site-data-and-paths.md) <br>
- [validation-evidence.md](references/validation-evidence.md) <br>
- [NVIDIA FLARE Documentation](https://nvflare.readthedocs.io/en/main) <br>
- [NVIDIA FLARE Paper](https://arxiv.org/abs/2210.13291) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Configuration instructions] <br>
**Output Format:** [Markdown] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 3 internal evaluation tasks (3 positive) across 2 agents, with 3 attempts per task in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use: checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed: skill selection, decoy avoidance, and workflow execution. <br>
- Effectiveness: Whether the skill helped complete the user's goal (50% goal completion + 50% expected workflow adherence). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage (50% tool-call productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 85.3% | 78.3% |
| Security | 100.0% → 100.0% (±0.0 points) | 80.0% → 66.7% (-13.3 points) |
| Correctness | 86.7% → 100.0% (+13.3 points) | 76.0% → 100.0% (+24.0 points) |
| Discoverability | 71.7% | 66.7% |
| Effectiveness | 72.5% → 81.7% (+9.2 points) | 48.0% → 80.0% (+32.0 points) |
| Efficiency | 72.9% | 78.3% |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
