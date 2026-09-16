## Description: <br>
Used for extracting selected metadata from one DICOM file and flagging standard-tag PHI presence. Not for anonymization or clinical use. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to extract selected DICOM metadata and flag standard-tag PHI presence before sharing or processing medical imaging files externally. <br>

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
- [Output JSON Schema](validators/output_schema.json) <br>
- [Agent Guide](AGENTS.md) <br>
- [Skill Benchmark](BENCHMARK.md) <br>
- [Skill Manifest](skill_manifest.yaml) <br>


## Skill Output: <br>
**Output Type(s):** [JSON, Analysis] <br>
**Output Format:** [JSON] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 2 evaluation tasks (2 positive) with 3 attempts per task in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use — checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the extracted metadata answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded and activated when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and expected workflow (equal-weight mean of goal completion and behavior adherence). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage (50% tool-call productivity, 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 72.3% — baseline ran, but no comparable score was available; uplift unavailable | 77.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 50.0% (-50.0 points) | 100.0% → 50.0% (-50.0 points) |
| Correctness | 25.0% → 80.0% (+55.0 points) | 15.0% → 100.0% (+85.0 points) |
| Discoverability | 91.5% — baseline ran, but no comparable score was available; uplift unavailable | 75.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 28.3% → 55.0% (+26.7 points) | 28.3% → 85.0% (+56.7 points) |
| Efficiency | 84.8% — baseline ran, but no comparable score was available; uplift unavailable | 74.8% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
0.1.0 (source: skill_manifest.yaml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
