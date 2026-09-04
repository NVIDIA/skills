## Description: <br>
NVIDIA DeepStream SDK development skill providing guided code generation with Python pyservicemaker API for building video analytics pipelines, GStreamer-based video processing, TensorRT inference integration, object detection/tracking, and Kafka/message broker integration. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers building real-time video analytics pipelines with NVIDIA DeepStream SDK, including multi-stream inference, object detection/tracking, and message broker integration on NVIDIA GPUs. <br>

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
- [GStreamer Plugins Reference](references/gstreamer_plugins.md) <br>
- [Service Maker API](references/service_maker_api.md) <br>
- [Use Cases and Pipelines](references/use_cases_pipelines.md) <br>
- [Streaming Sources](references/streaming_sources.md) <br>
- [Kafka Messaging](references/kafka_messaging.md) <br>
- [Best Practices](references/best_practices.md) <br>
- [Buffer APIs](references/buffer_apis.md) <br>
- [Media Extractor Advanced](references/media_extractor_advanced.md) <br>
- [Utilities and Config](references/utilities_config.md) <br>
- [nvinfer Config Reference](references/nvinfer_config.md) <br>
- [Tracker Config](references/tracker_config.md) <br>
- [Troubleshooting](references/troubleshooting.md) <br>
- [REST API Dynamic Sources](references/rest_api_dynamic.md) <br>
- [Metamux Config](references/metamux_config.md) <br>
- [Docker Containers](references/docker_containers.md) <br>
- [NVDS Message API Adapter](references/nvds_msgapi_adapter.md) <br>
- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions, Shell commands] <br>
**Output Format:** [Markdown with inline Python and bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 7 positive evaluation tasks in isolated k8s-sandbox pods, 1 attempt per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Whether the user's goal was achieved and expected workflow behavior was followed. <br>
- Efficiency: Routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 63% → 83% (+20 points) | 65% → 82% (+18 points) |
| Security | 71% → 86% (+14 points) | 71% → 79% (+7 points) |
| Correctness | 86% → 89% (+3 points) | 94% → 94% (±0 points) |
| Discoverability | 41% → 78% (+37 points) | 40% → 69% (+29 points) |
| Effectiveness | 91% → 97% (+6 points) | 93% → 98% (+5 points) |
| Efficiency | 25% → 64% (+39 points) | 24% → 72% (+48 points) |

## Skill Version(s): <br>
1.1.1 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
