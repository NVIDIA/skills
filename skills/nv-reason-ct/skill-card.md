## Description: <br>
Run NV-Reason-CT inference on user-provided 3D NIfTI chest or abdominal CT volumes for engineering and research workflows. Not for diagnosis, treatment, or clinical reporting. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to run NV-Reason-CT inference on 3D NIfTI chest or abdominal CT volumes, producing structured JSON results with input geometry, anatomy-region selection, response text, and runtime metadata for engineering and research workflows. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Optional] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [NV-Reason-CT Installation Instructions](https://github.com/NVIDIA-Medtech/NV-Reason-CT#installation) <br>
- [EXAMPLES.md](EXAMPLES.md) <br>
- [BENCHMARK.md](BENCHMARK.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis] <br>
**Output Format:** [JSON] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
9 evaluation tasks (9 positive), each with 3 attempts per task in isolated sandbox pods. Dataset digest: sha256:df577b70bec14f79cd9d434a2db86fd8aaab1d9fda79314f3d42d6e9494e6ab9. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use, checking for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the answer is correct, measured by final-answer accuracy against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed, including selection, decoy avoidance, and workflow execution. <br>
- Effectiveness: Whether the skill helped complete the task, combining goal completion (50%) and expected workflow adherence (50%). <br>
- Efficiency: Whether wasted tool calls and token usage were avoided, combining tool-call productivity (50%) and token efficiency (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 89.1% | 86.8% |
| Security | 76.7% → 90.0% (+13.3 pp) | 76.9% → 88.9% (+12.0 pp) |
| Correctness | 41.3% → 96.0% (+54.7 pp) | 52.3% → 95.6% (+43.3 pp) |
| Discoverability | 85.8% | 83.9% |
| Effectiveness | 52.6% → 87.0% (+34.4 pp) | 51.7% → 83.2% (+31.5 pp) |
| Efficiency | 86.8% | 82.3% |

## Skill Version(s): <br>
98e6322 (source: git SHA, committed 2026-09-24) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
