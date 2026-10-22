## Description: <br>
Guides raw verbs usage inside DOCA Core (doca-verbs) — direct control of QP, CQ, PD, SRQ and address-handle objects, porting libibverbs code into DOCA Core, and diagnosing raw doca_verbs_* attributes, opcodes, or WR flags the higher-level doca-rdma task API does not expose. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
External developers building applications that consume the DOCA Verbs library use this skill to decide whether to drop from a higher-level DOCA library to raw doca_verbs_* control, and to configure, build, modify, run, test, and debug raw QP / CQ / PD / MR / SRQ / Address-Handle verbs inside a DOCA Core context. <br>

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
- [command-appendix.md](references/command-appendix.md) <br>
- [deferred-topics.md](references/deferred-topics.md) <br>
- [deferred-verbs.md](references/deferred-verbs.md) <br>
- [install.md](references/install.md) <br>
- [pattern-overview.md](references/pattern-overview.md) <br>
- [use.md](references/use.md) <br>
- [DOCA SDK documentation](https://docs.nvidia.com/doca/sdk/index.html) <br>
- [NVIDIA DOCA samples](https://github.com/NVIDIA-DOCA/doca-samples) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), one attempt per task, each run in its own isolated k8s-sandbox pod against dataset snapshot skill-evaluator-dataset-snapshot/1. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use — no unsafe operations, secret leakage, or unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed and decoys were avoided. <br>
- Effectiveness: Whether the skill helped complete the user's goal and the expected workflow was followed. <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 96.8% (uplift unavailable) | 97.8% (uplift unavailable) |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 35.0% → 100.0% (+65.0 points) | 70.0% → 100.0% (+30.0 points) |
| Discoverability | 100.0% (uplift unavailable) | 95.0% (uplift unavailable) |
| Effectiveness | 21.3% → 98.1% (+76.8 points) | 58.8% → 98.8% (+40.0 points) |
| Efficiency | 86.0% (uplift unavailable) | 95.3% (uplift unavailable) |

## Skill Version(s): <br>
946399d (source: git SHA, committed 2026-10-21) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
