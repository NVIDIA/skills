## Description: <br>
Recommend NVSHMEM NIC-to-PE mappings and environment exports for HCA selection, multi-NIC configuration, or topology-based mapping diagnostics. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to obtain topology-informed NVSHMEM NIC-to-PE mapping recommendations and exact environment variable exports for multi-GPU HPC nodes. <br>

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
- [Version/Transport Matrix](references/version-transport-matrix.md) <br>
- [Topology and Output Contract](references/topology-and-output.md) <br>
- [NVSHMEM API Documentation](https://docs.nvidia.com/nvshmem/api/) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration instructions, Shell commands] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
18 evaluation tasks (17 positive, 1 negative) across isolated sandbox pods, with 3 attempts per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was selected and the workflow executed. <br>
- Effectiveness: Whether the user's goal was achieved and the expected workflow behavior was followed. <br>
- Efficiency: Tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 90.4% | 88.4% |
| Security | 96.6% → 100.0% (+3.4 pp) | 80.4% → 100.0% (+19.6 pp) |
| Correctness | 57.9% → 97.8% (+39.9 pp) | 72.9% → 92.2% (+19.3 pp) |
| Discoverability | 87.9% | 76.8% |
| Effectiveness | 37.8% → 73.8% (+36.0 pp) | 44.0% → 78.5% (+34.5 pp) |
| Efficiency | 92.2% | 94.4% |

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
