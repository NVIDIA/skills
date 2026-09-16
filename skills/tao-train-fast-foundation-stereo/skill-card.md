## Description: <br>
Real-time stereo depth estimation using FastFoundationStereo (FFS), the distilled bp2 commercial variant of FoundationStereo, predicting disparity maps from stereo image pairs with ~10× lower latency than full FoundationStereo. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers training, evaluating, exporting, or running inference for real-time stereo depth estimation models using NVIDIA TAO FastFoundationStereo. <br>

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
- [Setup and Run Guide](references/setup-and-run.md) <br>
- [Important Parameters](references/important-parameters.md) <br>
- [Error Patterns](references/error-patterns.md) <br>
- [Spec Overrides](references/spec-overrides.md) <br>
- [Parent Model Inference](references/parent-model-inference.md) <br>
- [TAO Deploy Fast Foundation Stereo](references/tao-deploy-fast-foundation-stereo.md) <br>
- [NVIDIA TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>
- [Agent Skills Open Standard](https://agentskills.io) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
1 evaluation task (1 positive) run in isolated k8s-sandbox pods with evaluator v1.3.2. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
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
| Overall | 27% → 100% (+73 points) | 33% → 39% (+6 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 20% → 100% (+80 points) | 20% → 40% (+20 points) |
| Discoverability | 0% → 100% (+100 points) | 0% → 0% (±0 points) |
| Effectiveness | 17% → 100% (+83 points) | 43% → 53% (+10 points) |
| Efficiency | 0% → 100% (+100 points) | 0% → 0% (±0 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
