## Description: <br>
Deploys and verifies the CDF Orchestration Airflow controller on Kubernetes — covering make setup, controller image builds, Helm installation, controller connection, port-forwarding, and post-install verification — after the cdf-orchestration-preflight audit confirms the checkout and target environment are ready. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to deploy, reinstall, upgrade, or connect to the CDF Orchestration Airflow controller on an existing Kubernetes cluster and to run post-install verification before handing off to a workflow. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key, Cloud Credentials, Other [kubeconfig / Kubernetes cluster access]] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [controller-connection.md](references/controller-connection.md) <br>
- [deploy-controller.md](references/deploy-controller.md) <br>
- [model-startup-timeouts.md](references/model-startup-timeouts.md) <br>
- [topologies.md](references/topologies.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Produces a deployment report summarizing preflight used, installation-readiness result, topology, commands executed, controller image tag, Airflow URL, controller checks, and unresolved warnings.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
11 evaluation tasks (9 positive, 2 negative), 1 attempt per task, each run in its own isolated k8s-sandbox pod (dataset digest sha256:4387b9f4a4d9ac65bee1d14597318ac2da37624b65c0d788a1f0689f00881a44). <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct against the reference answer? <br>
- Discoverability: Was the right skill loaded and activated when needed, avoiding decoys? <br>
- Effectiveness: Did the skill help complete the task, measured as the equal-weight mean of goal completion and expected workflow adherence? <br>
- Efficiency: Did it avoid wasted tool calls and token usage (tool productivity plus token efficiency)? <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 85.8% (uplift unavailable) | 70.6% (uplift unavailable) |
| Security | 90.9% → 95.5% (+4.6 points) | 54.6% → 77.3% (+22.7 points) |
| Correctness | 16.4% → 85.5% (+69.1 points) | 16.4% → 74.6% (+58.2 points) |
| Discoverability | 91.4% (uplift unavailable) | 72.2% (uplift unavailable) |
| Effectiveness | 24.9% → 62.7% (+37.8 points) | 19.9% → 45.8% (+25.9 points) |
| Efficiency | 93.9% (uplift unavailable) | 83.3% (uplift unavailable) |

## Skill Version(s): <br>
1.1.3 (source: SKILL.md frontmatter; repo pyproject.toml reports 1.1.0) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
