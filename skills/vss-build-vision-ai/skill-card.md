## Description: <br>
Adds agent-ready vision capabilities (dense captioning, detection, search, alerting, summarization) to an agent or application by routing, composing, configuring, and deploying a customizable, self-contained vision stack built on the NVIDIA VSS Blueprint, either as a stock base, alerts, LVS, or search developer profile, the warehouse industry profile, or a lean custom combination expressed as a delta overlay. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and agents who want to give their application vision use this skill to pick capabilities through guided intake or natural language, then compose and deploy the matching VSS Docker Compose stack. It also covers lifecycle changes such as installing an agent harness, resizing, or tearing the build down. <br>

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
- [composition.md](references/composition.md) <br>
- [deployment.md](references/deployment.md) <br>
- [agent-harness.md](references/agent-harness.md) <br>
- [deployment_resolution.md](references/deployment_resolution.md) <br>
- [teardown.md](references/teardown.md) <br>
- [prerequisites.md](references/prerequisites.md) <br>
- [credentials.md](references/credentials.md) <br>
- [ngc.md](references/ngc.md) <br>
- [sizing.md](references/sizing.md) <br>
- [edge.md](references/edge.md) <br>
- [readiness.md](references/readiness.md) <br>
- [troubleshooting.md](references/troubleshooting.md) <br>
- [profiles/](references/profiles/) <br>
- [services/](references/services/) <br>
- [NVIDIA VSS documentation](https://docs.nvidia.com/vss/latest/index.html) <br>
- [VSS Blueprint on GitHub](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) <br>
- [VSS Blueprint on build.nvidia.com](https://build.nvidia.com/nvidia/video-search-and-summarization) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Files] <br>
**Output Format:** [Markdown with inline bash code blocks, plus generated build artifacts (Compose overlays, env files, resolved.yml)] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Build artifacts are written under _builds/<name>/; credential values are never printed] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
11 evaluation tasks (10 positive, 1 negative), one attempt per task, each run in an isolated k8s-sandbox pod and compared against a no-skill baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the security signal. <br>
- Correctness: Is the answer correct? Scored from final-answer accuracy. <br>
- Discoverability: Was the right skill loaded when needed? Scored from skill execution. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal accuracy and behavior check. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight mean of tool productivity and token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 73.7% — baseline ran, but no comparable score was available; uplift unavailable | 59.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 72.7% → 90.9% (+18.2 points) | 40.9% → 50.0% (+9.1 points) |
| Correctness | 14.6% → 70.9% (+56.3 points) | 38.2% → 70.9% (+32.7 points) |
| Discoverability | 99.8% — baseline ran, but no comparable score was available; uplift unavailable | 80.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 6.7% → 27.5% (+20.8 points) | 14.7% → 21.0% (+6.3 points) |
| Efficiency | 79.5% — baseline ran, but no comparable score was available; uplift unavailable | 74.6% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
3.3.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
