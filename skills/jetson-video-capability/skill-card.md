## Description: <br>
Use when Jetson codec, profile, chroma, bit-depth, dimension, engine-count, or operational support must be reconciled using live SDK APIs, authenticated NVIDIA samples, and NVIDIA documentation. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers working with NVIDIA Jetson devices who need to determine video codec support, profile availability, and operational capability using live SDK APIs and NVIDIA documentation. <br>

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
- [Capability Queries](references/capability-queries.md) <br>
- [Surface Selection Contract](references/surface-selection-contract.md) <br>
- [Agent Skills](https://agentskills.io/) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (4 positive), each in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and followed expected workflow behavior. <br>
- Efficiency: Whether the skill avoided wasted tool or skill usage and maintained routing quality. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 59% → 93% (+33 points) | 68% → 84% (+16 points) |
| Security | 100% → 100% (±0 points) | 100% → 75% (-25 points) |
| Correctness | 80% → 100% (+20 points) | 100% → 100% (±0 points) |
| Discoverability | 37% → 98% (+62 points) | 50% → 91% (+41 points) |
| Effectiveness | 53% → 74% (+20 points) | 75% → 58% (-17 points) |
| Efficiency | 27% → 91% (+64 points) | 18% → 98% (+80 points) |

## Skill Version(s): <br>
e61c045 (source: git SHA, committed 2026-08-10) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
