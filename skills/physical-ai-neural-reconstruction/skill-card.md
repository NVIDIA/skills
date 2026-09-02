## Description: <br>
Router for NVIDIA NuRec/NRE: USDZ rendering, NCore conversion, 3DGS, gRPC sensor sim, carline adaptation, PhysicalAI HF datasets. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers working with NVIDIA Physical AI workflows who need to route neural reconstruction requests to the correct upstream NuRec sibling skill for USDZ rendering, NCore conversion, 3D Gaussian Splatting, sensor simulation, dataset management, and object harvesting. <br>

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
- [NVIDIA NuRec Skills (upstream)](https://github.com/NVIDIA/nurec-skills) <br>
- [NVIDIA NCore](https://github.com/NVIDIA/ncore) <br>
- [NVIDIA Asset Harvester](https://github.com/NVIDIA/asset-harvester) <br>
- [NVIDIA Harmonizer (DiffusionHarmonizer)](https://github.com/NVIDIA/harmonizer) <br>
- [NVIDIA PhysicalAI Datasets on Hugging Face](https://huggingface.co/nvidia) <br>
- [Workflows reference](references/workflows.md) <br>
- [Mix-ups and naming overlaps](references/mix-ups.md) <br>
- [Upstream fetch recipe](references/upstream-fetch.md) <br>
- [Maintenance guide](references/maintenance.md) <br>
- [Teardown guide](references/teardown.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative) run in isolated sandbox pods, evaluated against the skill-evaluator-dataset-snapshot/1 dataset. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal (goal completion and expected workflow adherence, equally weighted). <br>
- Efficiency: Checks routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Detects unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies whether the expected skill was found and executed. <br>
- `skill_efficiency`: Measures routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Measures final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Measures whether the user's goal was achieved. <br>
- `behavior_check`: Verifies whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 55% → 93% (+38 points) | 66% → 87% (+21 points) |
| Security | 100% → 100% (±0 points) | 100% → 100% (±0 points) |
| Correctness | 35% → 95% (+60 points) | 90% → 90% (±0 points) |
| Discoverability | 62% → 100% (+38 points) | 62% → 92% (+30 points) |
| Effectiveness | 26% → 78% (+51 points) | 44% → 61% (+17 points) |
| Efficiency | 53% → 95% (+42 points) | 34% → 92% (+58 points) |

## Skill Version(s): <br>
0.4.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
