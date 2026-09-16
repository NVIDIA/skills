## Description: <br>
Router for NVIDIA NuRec/NRE: USDZ rendering, NCore conversion, 3DGS, gRPC sensor sim, carline adaptation, PhysicalAI HF datasets. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers working with NVIDIA Neural Reconstruction (NuRec) workflows use this skill to identify the correct upstream sibling skill for data conversion, 3D scene training, rendering, object extraction, and frame cleanup. <br>

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
- [what-is-nurec.md](references/what-is-nurec.md) <br>
- [workflows.md](references/workflows.md) <br>
- [prerequisites.md](references/prerequisites.md) <br>
- [upstream-fetch.md](references/upstream-fetch.md) <br>
- [troubleshooting.md](references/troubleshooting.md) <br>
- [mix-ups.md](references/mix-ups.md) <br>
- [maintenance.md](references/maintenance.md) <br>
- [teardown.md](references/teardown.md) <br>
- [NVIDIA NuRec Skills (GitHub)](https://github.com/NVIDIA/nurec-skills) <br>
- [NVIDIA Asset Harvester (GitHub)](https://github.com/NVIDIA/asset-harvester) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), each attempt in an isolated sandbox pod. Dataset digest: sha256:be81cdb0740b0f4d42cef1ea2512c5ce39842d5218a9e6cb9d87a601834fadc4. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was selected and activated when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal (50% goal completion + 50% expected workflow adherence). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage (50% tool-call productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (routing scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 93.8% | 89.6% |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 33.3% → 95.0% (+61.7 points) | 95.0% → 100.0% (+5.0 points) |
| Discoverability | 100.0% | 95.0% |
| Effectiveness | 22.2% → 79.4% (+57.2 points) | 45.0% → 69.4% (+24.4 points) |
| Efficiency | 94.4% | 83.5% |

## Skill Version(s): <br>
0.4.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
