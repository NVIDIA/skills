## Description: <br>
Add agent-ready vision capabilities — dense captioning, detection, search, alerting, summarization — to an agent or application through a customizable, self-contained vision stack built on the NVIDIA VSS Blueprint. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers who want to give an agent or application vision capabilities use this skill to route, compose, configure, and deploy a self-contained VSS vision stack — a stock developer profile (base, alerts, lvs, search), the warehouse industry profile, or a lean custom delta overlay — via Docker Compose. <br>

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
- [composition.md — delta-profile rules and build artifact contract](references/composition.md) <br>
- [deployment.md — resolved Compose deployment lifecycle](references/deployment.md) <br>
- [agent-harness.md — in-stack agent and NemoClaw harness contract](references/agent-harness.md) <br>
- [deployment_resolution.md — public-route mappings and endpoint contract](references/deployment_resolution.md) <br>
- [teardown.md — build and harness destruction lifecycle](references/teardown.md) <br>
- [prerequisites.md — host, GPU runtime, firewall, and NGC checks](references/prerequisites.md) <br>
- [sizing.md — developer-profile sizing and model placement](references/sizing.md) <br>
- [edge.md — DGX Spark and Thor routing and edge model recipes](references/edge.md) <br>
- [profiles/ — developer profile capabilities and service sets](references/profiles/) <br>
- [services/ — capability-owner service contracts](references/services/) <br>
- [NVIDIA AI Blueprint: Video Search and Summarization (demo)](https://build.nvidia.com/nvidia/video-search-and-summarization) <br>
- [VSS Documentation](https://docs.nvidia.com/vss/latest/index.html) <br>
- [video-search-and-summarization (GitHub)](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Code] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Generates Docker Compose build artifacts under _builds/<name>/ (override.env, compose.yml, resolved.yml)] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
11 evaluation tasks (10 positive, 1 negative), one attempt per task, each run in its own isolated k8s-sandbox pod against dataset snapshot skill-evaluator-dataset-snapshot/1. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct? Final-answer correctness against the reference answer. <br>
- Discoverability: Was the right skill loaded when needed? Whether the expected skill was selected and decoys avoided. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight mean of tool-call productivity and token efficiency. <br>

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
| Overall | 67.9% — uplift unavailable | 60.5% — uplift unavailable |
| Security | 72.7% → 59.1% (-13.6 points) | 27.3% → 63.6% (+36.3 points) |
| Correctness | 27.3% → 67.3% (+40.0 points) | 47.3% → 50.9% (+3.6 points) |
| Discoverability | 99.3% — uplift unavailable | 77.5% — uplift unavailable |
| Effectiveness | 3.8% → 30.2% (+26.4 points) | 15.9% → 28.7% (+12.8 points) |
| Efficiency | 83.6% — uplift unavailable | 81.6% — uplift unavailable |

## Skill Version(s): <br>
3.3.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
