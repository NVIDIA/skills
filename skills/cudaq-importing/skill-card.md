## Description: <br>
Use when porting circuits from another framework (e.g. Qiskit) into CUDA-Q kernels while preserving the source algorithm and validation fidelity. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to port quantum circuits from other frameworks (e.g., Qiskit) into CUDA-Q Python kernels while preserving algorithm correctness and validation fidelity. <br>

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
- [Qiskit to CUDA-Q Porting Reference](references/porting-reference.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions] <br>
**Output Format:** [Markdown with inline Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), 3 attempts each, in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use: checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Whether the skill helped complete the user's goal (50% goal accuracy + 50% expected workflow adherence). <br>
- Efficiency: Tool-call productivity and token efficiency (50% each). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (routing scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 91.9% | 94.9% |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 90.0% → 100.0% (+10.0 points) | 90.0% → 100.0% (+10.0 points) |
| Discoverability | 93.3% | 95.0% |
| Effectiveness | 79.2% → 89.8% (+10.6 points) | 76.7% → 93.1% (+16.4 points) |
| Efficiency | 76.3% | 86.6% |

## Skill Version(s): <br>
1.0.2 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
