## Description: <br>
Run the disk-backed DEFT AOI improvement loop for NVIDIA Cosmos Reason 3 / Cosmos3 models, using Nano by default and Edge or Super when explicitly requested: evaluate the base model on Proxy and frozen Benchmark splits, mine real image pairs from Proxy gaps, generate AnomalyGen synthetic NG pairs, assemble a per-iteration Train JSON from both producers, train with Cosmos Framework LoRA SFT, and repeat through the selected platform's submit/status/logs/cancel contract, supporting bare OK/NG labels only. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 AND CC-BY-4.0 <br>
## Use Case: <br>
Developers and engineers use this skill to iteratively improve NVIDIA Cosmos Reason 3 model accuracy on bare OK/NG PCB AOI (Automated Optical Inspection) tasks through automated evaluation, real-image mining, synthetic anomaly generation, training-data assembly, and Cosmos Framework LoRA SFT training. <br>

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
- [air-gap.md](references/air-gap.md) <br>
- [network-bootstrap.md](references/network-bootstrap.md) <br>
- [metric-contract.md](references/metric-contract.md) <br>
- [gap-analysis.md](references/gap-analysis.md) <br>
- [aoi-annotation.md](references/aoi-annotation.md) <br>
- [scripts-and-agents.md](references/scripts-and-agents.md) <br>
- [tao-generate-anomalies.md](references/tao-generate-anomalies.md) <br>
- [tao-mine-aoi-images.md](references/tao-mine-aoi-images.md) <br>
- [REPORT_RENDERING.md](references/REPORT_RENDERING.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Files, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 3 evaluation tasks (3 positive), 1 attempt per task, each in its own isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal. <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal. <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% `skill_efficiency` plus 50% `token_efficiency`. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 92.0% — baseline ran, but no comparable score was available; uplift unavailable | 85.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 66.7% → 100.0% (+33.3 points) | 66.7% → 100.0% (+33.3 points) |
| Correctness | 6.7% → 100.0% (+93.3 points) | 26.7% → 73.3% (+46.6 points) |
| Discoverability | 98.3% — baseline ran, but no comparable score was available; uplift unavailable | 91.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 9.2% → 68.3% (+59.1 points) | 12.5% → 65.8% (+53.3 points) |
| Efficiency | 93.5% — baseline ran, but no comparable score was available; uplift unavailable | 97.2% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
0.2.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
