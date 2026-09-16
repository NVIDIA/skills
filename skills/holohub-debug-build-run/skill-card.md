## Description: <br>
Use when a concrete ./holohub command fails, hangs, regresses, or returns wrong output and needs reproducible diagnosis and verification. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers debugging concrete HoloHub CLI command failures, hangs, regressions, or incorrect output through reproducible diagnosis and minimal fixes with regression proof. <br>

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
- [HoloHub CLI contract](references/holohub-cli-contract.md) <br>
- [Debug workflow](references/debug-workflow.md) <br>
- [Version-sensitive diagnostic priors](references/known-issues.md) <br>
- [NVIDIA HoloHub (GitHub)](https://github.com/nvidia-holoscan/holohub) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Code] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (2 positive, 2 negative), each in an isolated sandbox pod. Evaluator version 1.3.2. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the skill produces correct answers against the reference answer. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helps the agent complete the user's goal and follow the expected workflow. <br>
- Efficiency: Whether the skill avoids wasted tool or skill usage through quality routing and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 64% → 89% (+25 points) | 60% → 69% (+9 points) |
| Security | 100% → 100% (±0 points) | 75% → 100% (+25 points) |
| Correctness | 45% → 85% (+40 points) | 60% → 60% (±0 points) |
| Discoverability | 61% → 100% (+39 points) | 56% → 70% (+14 points) |
| Effectiveness | 55% → 60% (+6 points) | 48% → 40% (-9 points) |
| Efficiency | 58% → 99% (+41 points) | 58% → 75% (+17 points) |

## Skill Version(s): <br>
09a1eb2d (source: git SHA, committed 2026-09-02) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
