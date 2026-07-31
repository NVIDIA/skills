## Description: <br>
Use when a user asks to build, optimize, backtest, rebalance, or analyze a stock portfolio with Mean-CVaR, Mean-Variance/SOCP variance caps, efficient frontiers, scenario generation, or NVIDIA cuOpt. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and quantitative finance engineers use this skill to build, optimize, backtest, and rebalance stock portfolios using NVIDIA GPU-accelerated Mean-CVaR and Mean-Variance optimization with cuOpt. <br>

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
- [Agent Recipes (workflow reference)](references/workflows/agent_recipes.md) <br>
- [NVIDIA cuOpt](https://github.com/NVIDIA/cuopt) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Analysis, Files] <br>
**Output Format:** [Markdown with inline Python code blocks, tables, and figures] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 4 tasks (2 positive, 2 negative) in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal and followed expected workflow behavior. <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Verifies absence of unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `skill_execution`: Verifies whether the expected skill was found and executed. <br>
- `goal_accuracy`: Verifies whether the user's goal was achieved. <br>
- `behavior_check`: Verifies whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Verifies routing quality, workspace-aware skill reads, and productive tool use. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 64% → 84% (+20 points) | 58% → 72% (+14 points) |
| Security | 75% → 100% (+25 points) | 75% → 75% (±0 points) |
| Correctness | 60% → 85% (+25 points) | 60% → 70% (+10 points) |
| Discoverability | 69% → 88% (+19 points) | 64% → 88% (+23 points) |
| Effectiveness | 50% → 52% (+2 points) | 33% → 41% (+8 points) |
| Efficiency | 66% → 96% (+30 points) | 60% → 89% (+29 points) |

## Skill Version(s): <br>
08ae570 (source: git SHA, committed 2026-07-29) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
