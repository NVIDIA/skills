## Description: <br>
Used for header-only preflight of one DICOM series folder before conversion or inference. Not for de-identification or clinical clearance. <br>

This skill is for research and development only. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to run a header-only preflight scan of a DICOM series directory, checking orientation, spacing consistency, and PHI-tag presence before conversion or model inference. <br>

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
- [Skill Manifest](skill_manifest.yaml) <br>
- [Skill Benchmark](BENCHMARK.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Files] <br>
**Output Format:** [JSON] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Emits structured preflight JSON with inventory, orientation axcodes, PHI flags, findings, and a verdict of pass, warn, or fail] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
3 evaluation tasks across DICOM preflight scenarios (3 positive cases), each with 3 attempts per agent in isolated sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use: checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded and activated when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal, combining goal completion (50%) and expected workflow adherence (50%). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage, combining tool-call productivity (50%) and token efficiency (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 75.3% | 71.6% |
| Security | 100.0% → 66.7% (-33.3 pts) | 100.0% → 66.7% (-33.3 pts) |
| Correctness | 2.9% → 80.0% (+77.1 pts) | 2.5% → 80.0% (+77.5 pts) |
| Discoverability | 88.3% | 81.7% |
| Effectiveness | 13.6% → 56.1% (+42.5 pts) | 17.5% → 49.7% (+32.2 pts) |
| Efficiency | 85.2% | 79.8% |

## Skill Version(s): <br>
0.1.0 (source: skill_manifest.yaml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
