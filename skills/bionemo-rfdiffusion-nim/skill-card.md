## Description: <br>
Run RFDiffusion protein backbone design via NVIDIA NIM for de novo protein backbones, motif scaffolding, binder design, and PDB backbone outputs for downstream ProteinMPNN sequence design. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and computational biologists use this skill to design protein backbone structures — de novo, motif scaffolding, or binder design — via the NVIDIA RFDiffusion NIM API (hosted or local Docker). <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [RFDiffusion NIM — API Reference](references/api.md) <br>
- [RFDiffusion Examples](references/examples.md) <br>
- [RFDiffusion Parameter Guidance](references/parameters.md) <br>
- [RFDiffusion Science Notes](references/science.md) <br>
- [RFDiffusion Validation](references/validation.md) <br>


## Skill Output: <br>
**Output Type(s):** [API Calls, Files] <br>
**Output Format:** [PDB text files and JSON API responses] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
1 evaluation task (1 positive case), 3 attempts per task, each in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was selected and the workflow executed. <br>
- Effectiveness: Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Tool-call productivity (50%) and token efficiency (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 67.8% | 90.6% |
| Security | 100.0% → 0.0% (-100.0 pts) | 50.0% → 100.0% (+50.0 pts) |
| Correctness | 100.0% → 100.0% (±0.0 pts) | 0.0% → 100.0% (+100.0 pts) |
| Discoverability | 100.0% | 95.0% |
| Effectiveness | 100.0% → 65.0% (-35.0 pts) | 100.0% → 92.9% (-7.1 pts) |
| Efficiency | 73.9% | 65.4% |

## Skill Version(s): <br>
0.1.0 (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
