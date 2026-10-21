## Description: <br>
Set up missing Linux prerequisites and audit whether a Cosmos Data Factory (CDF) Orchestration checkout and its Kubernetes or OSMO environment are ready to install the controller and run every packaged DAG. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers preparing a Cosmos Data Factory (CDF) Orchestration deployment use this skill to install missing Linux prerequisites and run a read-only audit that reports install and run readiness for the controller and every packaged DAG across Kubernetes or OSMO environments. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key, Other [kubeconfig file]] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [bootstrap.md](references/bootstrap.md) <br>
- [nfs-remediation.md](references/nfs-remediation.md) <br>
- [readiness-model.md](references/readiness-model.md) <br>
- [Docker Engine installation](https://docs.docker.com/engine/install/) <br>
- [Kubernetes container runtimes](https://kubernetes.io/docs/setup/production-environment/container-runtimes/) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks; optional machine-readable JSON report via --format json] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Read-only audit reporting independent install_readiness and run_readiness gates; prints bucket prefixes only and never emits credentials or secret bodies] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
14 evaluation tasks (10 positive, 4 negative), each attempt run in its own isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct? Measures final-answer correctness against the reference answer. <br>
- Discoverability: Was the right skill loaded when needed? Checks skill selection, decoy avoidance, and workflow execution. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Combines tool-call productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 86.8% (uplift unavailable) | 79.1% (uplift unavailable) |
| Security | 78.6% → 96.4% (+17.8 points) | 67.9% → 89.3% (+21.4 points) |
| Correctness | 12.9% → 80.0% (+67.1 points) | 25.7% → 68.6% (+42.9 points) |
| Discoverability | 99.0% (uplift unavailable) | 86.0% (uplift unavailable) |
| Effectiveness | 36.1% → 68.8% (+32.7 points) | 36.8% → 65.2% (+28.4 points) |
| Efficiency | 89.7% (uplift unavailable) | 86.5% (uplift unavailable) |

## Skill Version(s): <br>
1.1.2 (source: frontmatter; pyproject.toml reports 1.1.0) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
