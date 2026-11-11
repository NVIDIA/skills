## Description: <br>
Router for NVIDIA NuRec/NRE: USDZ rendering, NCore conversion, 3DGS, gRPC sensor sim, carline adaptation, PhysicalAI HF datasets. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers working with NVIDIA Neural Reconstruction (NuRec) workflows use this skill to identify and order the correct upstream sibling skill for dataset download, NCore data conversion, 3D scene training, rendering, object extraction, and frame cleanup. <br>

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
- [NVIDIA NuRec (GitHub)](https://github.com/NVIDIA/nurec) <br>
- [NVIDIA NCore (GitHub)](https://github.com/NVIDIA/ncore) <br>
- [NVIDIA Harmonizer (GitHub)](https://github.com/NVIDIA/harmonizer) <br>
- [NVIDIA Asset Harvester (GitHub)](https://github.com/NVIDIA/asset-harvester) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (3 positive, 1 negative), one attempt per task, each in an isolated sandbox pod. Dataset digest: sha256:49c2655474f42be968d68e23c8bf6a745d5e19db4893ecb71cc769ac2ee5b636. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed. <br>
- Effectiveness: Whether the skill helped complete the task (50% goal completion + 50% expected workflow adherence). <br>
- Efficiency: Whether it avoided wasted tool calls and token usage (50% tool-call productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 89.4% | 79.6% |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 40.0% → 85.0% (+45.0 points) | 85.0% → 85.0% (±0.0 points) |
| Discoverability | 100.0% | 91.7% |
| Effectiveness | 20.0% → 64.4% (+44.4 points) | 39.4% → 49.4% (+10.0 points) |
| Efficiency | 97.7% | 72.0% |

## Skill Version(s): <br>
0.4.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
