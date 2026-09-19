## Description: <br>
Real-time stereo depth estimation using FastFoundationStereo (FFS), the distilled bp2 commercial variant of FoundationStereo that predicts disparity maps from rectified stereo image pairs with ~10× lower latency than full FoundationStereo. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to train, evaluate, export, and run inference for NVIDIA TAO FastFoundationStereo (FFS) stereo depth estimation models using Docker containers with NVIDIA Container Toolkit. <br>

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
- [setup-and-run.md](references/setup-and-run.md) <br>
- [important-parameters.md](references/important-parameters.md) <br>
- [error-patterns.md](references/error-patterns.md) <br>
- [spec-overrides.md](references/spec-overrides.md) <br>
- [parent-model-inference.md](references/parent-model-inference.md) <br>
- [tao-deploy-fast-foundation-stereo.md](references/tao-deploy-fast-foundation-stereo.md) <br>
- [skill_info.yaml](references/skill_info.yaml) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
1 evaluation task (1 positive), 3 attempts per task, evaluated in k8s-sandbox environment with isolated sandbox pods per attempt. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks whether the answer is correct against the reference answer. <br>
- Discoverability: Checks whether the right skill was loaded and activated when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal and expected workflow (equal-weight mean of goal completion and workflow adherence). <br>
- Efficiency: Checks whether the skill avoided wasted tool calls and token usage (50% tool-call productivity, 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Verifies absence of unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `skill_execution`: Verifies whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Verifies whether the user's goal was achieved. <br>
- `behavior_check`: Verifies whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Verifies tool-call productivity (routing is scored under Discoverability). <br>
- `token_efficiency`: Verifies actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 99.5% | 62.4% |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 13.3% → 100.0% (+86.7 points) | 20.0% → 60.0% (+40.0 points) |
| Discoverability | 100.0% | 0.0% |
| Effectiveness | 22.2% → 100.0% (+77.8 points) | 48.3% → 53.3% (+5.0 points) |
| Efficiency | 97.6% | 99.7% → 98.5% (-1.2 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
