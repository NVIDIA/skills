## Description: <br>
Use when NVSHMEM beginners want a tutorial-style overview on assessment, mental models, first C/C++ or Python NVSHMEM programs, compilation, launching, and next steps. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers learning NVSHMEM who need a tutorial-style onboarding covering fit assessment, mental models, first C/C++ or Python programs, compilation, and launch patterns for GPU-initiated communication. <br>

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
- [NVSHMEM Mental Model](references/mental-model.md) <br>
- [Basic API Calls](references/basic-api-calls.md) <br>
- [NVSHMEM4Py Programs](references/nvshmem4py-programs.md) <br>
- [NVSHMEM Introduction and Advantages](https://docs.nvidia.com/nvshmem/api/latest/introduction.html#advantages-of-nvshmem) <br>
- [NVSHMEM Documentation](https://docs.nvidia.com/nvshmem/api/) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
6 evaluation tasks (5 positive, 1 negative) run with 3 attempts per task in isolated sandbox pods. Dataset digest: sha256:d4b763e4ad4d346d9193abfe1de56a613a73b0a613ad065db2291289b139b7a0. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use, checking for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the answer produced by the skill-assisted agent is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed, the expected skill was selected, and decoys were avoided. <br>
- Effectiveness: Whether the skill helped complete the user's goal (50% goal completion + 50% expected workflow adherence). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage (50% tool-call productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity; routing is scored under Discoverability. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 87.5% | 89.1% |
| Security | 100.0% → 100.0% (±0.0 points) | 70.0% → 100.0% (+30.0 points) |
| Correctness | 47.5% → 86.7% (+39.2 points) | 66.0% → 100.0% (+34.0 points) |
| Discoverability | 80.0% | 73.0% |
| Effectiveness | 48.4% → 84.0% (+35.6 points) | 50.0% → 87.8% (+37.8 points) |
| Efficiency | 87.0% | 84.5% |

## Skill Version(s): <br>
0c1e40d1e (source: git SHA, committed 2026-09-17) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
