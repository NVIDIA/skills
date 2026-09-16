## Description: <br>
Scaffold a standalone RTVI CV microservice that plugs into VSS Search and Alerts profiles via Kafka mdx-raw, using a YOLO26 reference implementation. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
NVIDIA Proprietary <br>
## Use Case: <br>
Developers and engineers building custom perception microservices that integrate with NVIDIA VSS (Video Search and Summarization) deployments, replacing or augmenting the default perception service with a YOLO26-based detector while keeping downstream Search, Alerts, and Behavior Analytics workflows intact. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [VSS Integration Contract](references/integration-contract.md) <br>
- [VSS Profile Integration](references/vss-profile-integration.md) <br>
- [YOLO26 DeepStream Configuration](references/yolo26-deepstream.md) <br>
- [VSS Object Detection and Tracking](https://docs.nvidia.com/vss/latest/object-detection-tracking.html) <br>
- [VSS Behavior Analytics](https://docs.nvidia.com/vss/latest/behavior-analytics.html) <br>
- [VSS Search Workflow](https://docs.nvidia.com/vss/latest/agent-workflow-search.html) <br>
- [DeepStream Gst-nvmsgconv](https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_plugin_gst-nvmsgconv.html) <br>
- [DeepStream Gst-nvmsgbroker](https://docs.nvidia.com/metropolis/deepstream/9.1/text/DS_plugin_gst-nvmsgbroker.html) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Files, Shell commands, Configuration instructions] <br>
**Output Format:** [Scaffolded project directory with Dockerfile, Compose files, DeepStream configs, Python tests, and inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
3 evaluation tasks (3 positive), each in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and followed expected workflow behavior. <br>
- Efficiency: Routing quality, workspace-aware skill reads, and productive tool use without waste. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 58% → 96% (+38 points) | 47% → 85% (+38 points) |
| Security | 100% → 100% (±0 points) | 67% → 67% (±0 points) |
| Correctness | 60% → 100% (+40 points) | 47% → 100% (+53 points) |
| Discoverability | 50% → 96% (+46 points) | 40% → 81% (+42 points) |
| Effectiveness | 62% → 96% (+33 points) | 52% → 94% (+42 points) |
| Efficiency | 20% → 89% (+70 points) | 28% → 84% (+56 points) |

## Skill Version(s): <br>
2.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
