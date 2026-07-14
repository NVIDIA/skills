## Description: <br>
NVIDIA DeepStream SDK development with Python pyservicemaker API for building video analytics pipelines, GStreamer-based video processing, TensorRT inference integration, object detection/tracking, and Kafka/message broker integration. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers building NVIDIA DeepStream video analytics pipelines, GStreamer-based video processing applications, and TensorRT inference integrations using AI coding assistants. <br>

### Deployment Geography for Use: <br>
Global <br>

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
- [nvinfer Configuration](references/nvinfer_config.md) <br>
- [Tracker Configuration](references/tracker_config.md) <br>
- [Troubleshooting](references/troubleshooting.md) <br>
- [REST API Dynamic Sources](references/rest_api_dynamic.md) <br>
- [Docker Containers](references/docker_containers.md) <br>
- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) <br>
- [DeepStream NGC Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands, Configuration instructions] <br>
**Output Format:** [Python code and YAML/INI configuration files with inline documentation] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`claude-code`) <br>
- Codex (`codex`) <br>



## Evaluation Tasks: <br>
Evaluated against 7 tasks (5 positive skill-activation, 2 negative) in the NVSkills-Eval `external` profile on `astra-sandbox` environment. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies that the agent loaded the expected skill and workflow. <br>
- `skill_efficiency`: Checks routing quality, decoy avoidance, and redundant tool usage. <br>
- `accuracy`: Grades final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the overall user task completed successfully. <br>
- `behavior_check`: Verifies expected behavior steps, including safety expectations. <br>
- `token_efficiency`: Compares token usage with and without the skill. <br>



## Evaluation Results: <br>
| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 7 | 100% (+0%) | 93% (-7%) |
| Correctness | 7 | 77% (+7%) | 84% (+13%) |
| Discoverability | 7 | 68% (+21%) | 77% (+20%) |
| Effectiveness | 7 | 85% (+4%) | 84% (+10%) |
| Efficiency | 7 | 65% (+22%) | 65% (+13%) |

## Skill Version(s): <br>
1.1.1 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
