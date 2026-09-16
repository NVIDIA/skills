## Description: <br>
Use this skill for OpenFold3, NVIDIA's BioNeMo NIM microservice for biomolecular structure prediction. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and computational biologists who need to predict biomolecular structures — proteins, protein-ligand complexes, protein-DNA/RNA assemblies — using NVIDIA's hosted OpenFold3 API or a local Docker NIM deployment. <br>

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
- [API Reference](references/api.md) <br>
- [Examples](references/examples.md) <br>
- [Parameter Reference](references/parameters.md) <br>
- [Science Guide](references/science.md) <br>
- [Validation Guide](references/validation.md) <br>


## Skill Output: <br>
**Output Type(s):** [API Calls, Files, Analysis] <br>
**Output Format:** [Markdown with inline Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (4 positive), 3 attempts per task, each in an isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was selected, decoys avoided, and workflow executed. <br>
- Effectiveness: Checks whether the user's goal was achieved and expected workflow behavior was followed. <br>
- Efficiency: Checks tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Detects unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity; routing is scored under Discoverability. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 91.2% | 83.6% |
| Security | 50.0% → 87.5% (+37.5 points) | 30.0% → 75.0% (+45.0 points) |
| Correctness | 85.0% → 100.0% (+15.0 points) | 92.0% → 95.0% (+3.0 points) |
| Discoverability | 100.0% | 91.3% |
| Effectiveness | 71.0% → 88.4% (+17.4 points) | 61.2% → 77.0% (+15.8 points) |
| Efficiency | 79.8% | 79.9% |

## Skill Version(s): <br>
0.1.0 (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
