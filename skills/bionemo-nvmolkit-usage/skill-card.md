## Description: <br>
Use when writing or debugging nvMolKit Python code for GPU-accelerated RDKit fingerprints, similarity, conformers, clustering, and molecular searches. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and computational chemists use this skill to write and debug GPU-accelerated molecular processing code with nvMolKit, including fingerprinting, similarity searches, conformer generation, forcefield optimization, clustering, and substructure/MCS searches. <br>

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
- [Advanced Usage Reference](references/advanced-usage.md) <br>
- [NVIDIA nvMolKit Documentation](https://nvidia-bionemo.github.io/nvMolKit/) <br>
- [nvMolKit Changelog](https://nvidia-bionemo.github.io/nvMolKit/changelog.html) <br>
- [nvMolKit Examples (Jupyter notebooks)](https://github.com/NVIDIA-BioNeMo/nvMolKit/tree/main/examples) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
12 evaluation tasks (12 positive), each with 3 attempts per task in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct? Verifies final-answer correctness against the reference answer. <br>
- Discoverability: Was the right skill loaded when needed? Checks whether the expected skill was selected and the workflow executed. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% tool-call productivity and 50% token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (routing scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 91.1% | 92.3% |
| Security | 66.7% → 91.7% (+25.0 pts) | 100.0% → 100.0% (±0.0 pts) |
| Correctness | 95.0% → 100.0% (+5.0 pts) | 96.7% → 96.7% (±0.0 pts) |
| Discoverability | 91.7% | 95.0% |
| Effectiveness | 91.4% → 92.6% (+1.2 pts) | 82.9% → 85.4% (+2.5 pts) |
| Efficiency | 79.5% | 84.6% |

## Skill Version(s): <br>
0.6.0 (source: pyproject.toml, CHANGELOG, released 2026-08-13) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
