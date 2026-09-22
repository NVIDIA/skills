## Description: <br>
Run the disk-backed DEFT AOI improvement loop for NVIDIA Cosmos Reason 3 / Cosmos3 models, using Nano by default and Edge or Super when explicitly requested: evaluate the base model on Proxy and frozen Benchmark splits, mine real image pairs from Proxy gaps, assemble a per-iteration Train JSON from selected Mining samples, train with cosmos-rl LoRA SFT, and repeat through the selected platform's submit/status/logs/cancel contract. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and engineers use this skill to iteratively improve NVIDIA Cosmos Reason 3 model accuracy on PCB AOI (Automated Optical Inspection) tasks through automated evaluation, image mining, anomaly generation, data assembly, and LoRA SFT training. <br>

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
- [TAO Skill Bank Repository](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>
- [cosmos-reason.md](references/cosmos-reason.md) <br>
- [pipeline-and-state.md](references/pipeline-and-state.md) <br>
- [data-layout.md](references/data-layout.md) <br>
- [preflight.md](references/preflight.md) <br>
- [gap-analysis.md](references/gap-analysis.md) <br>
- [aoi-annotation.md](references/aoi-annotation.md) <br>
- [tao-generate-anomalies.md](references/tao-generate-anomalies.md) <br>
- [tao-mine-aoi-images.md](references/tao-mine-aoi-images.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 3 evaluation tasks (3 positive) in isolated sandbox pods with 3 attempts per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Verifies final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was selected and the workflow executed. <br>
- Effectiveness: Measures goal completion (50%) and expected workflow behavior adherence (50%). <br>
- Efficiency: Measures tool-call productivity (50%) and token efficiency (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Detects unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 88.9% | 89.5% |
| Security | 88.9% → 100.0% (+11.1 points) | 77.8% → 100.0% (+22.2 points) |
| Correctness | 13.3% → 86.7% (+73.4 points) | 26.7% → 80.0% (+53.3 points) |
| Discoverability | 98.3% | 91.7% |
| Effectiveness | 9.7% → 70.8% (+61.1 points) | 19.5% → 79.2% (+59.7 points) |
| Efficiency | 88.9% | 96.9% |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
