## Description: <br>
Expert cuTile programming assistant that writes high-performance GPU kernels using cuTile's tile-based programming model with proper validation and optimization, and supports deep agent orchestration for complex multi-kernel tasks. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to write, convert, debug, validate, and optimize cuTile GPU kernels for NVIDIA GPUs, including multi-kernel decomposition of PyTorch modules via an orchestrated sub-agent workflow. <br>

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
- [cuTile Language Specification](https://docs.nvidia.com/cuda/cutile-python) <br>
- [Implementation Lessons](guidelines/01_implementation_lessons.md) <br>
- [Code Generation Rules](guidelines/02_code_generation_rules.md) <br>
- [Core Concepts](guidelines/03_concepts.md) <br>
- [TileGym and Examples Guide](examples/tilegym_and_examples_guide.md) <br>
- [Orchestration Overview](orchestration/overview.md) <br>
- [Orchestration Workflow](orchestration/workflow.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands] <br>
**Output Format:** [Single Python source file containing cuTile kernel, inline validation, and test code] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
3 evaluation tasks (2 positive, 1 negative), 1 attempt per task, each run in an isolated k8s-sandbox pod (Skill Evaluator 1.5.6, evaluated 2026-10-16). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal. <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal. <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% `skill_efficiency` plus 50% `token_efficiency`. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 91.8% — baseline ran, but no comparable score was available; uplift unavailable | 91.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Discoverability | 87.5% — baseline ran, but no comparable score was available; uplift unavailable | 88.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 75.8% → 91.1% (+15.3 points) | 84.8% → 84.5% (-0.3 points) |
| Efficiency | 80.3% — baseline ran, but no comparable score was available; uplift unavailable | 87.1% — baseline ran, but no comparable score was available; uplift unavailable |

Overall verdict: FAIL — Publication blocked (Tier 1 static validation FAILED; Tier 2 PASSED WITH OBSERVATIONS; Tier 3 live agent evaluation PASS).

## Skill Version(s): <br>
1.3.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
