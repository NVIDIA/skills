## Description: <br>
Run and operate the DeepStream Multi-View 3D Tracking reference app, also known as MV3DT. Use when the user asks to set up prerequisites, run shipped MV3DT samples, run Multi-View 3D Tracking on custom synchronized MP4 datasets, import camera calibration, delegate missing calibration to AutoMagicCalib, inspect OSD or BEV visualization, consume MV3DT Kafka metadata, or clean up MV3DT run state in the DeepStream MV3DT app directory. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to set up, run, verify, visualize, and tear down the DeepStream Multi-View 3D Tracking (MV3DT) reference app in the DeepStream container, on the shipped 4-camera and 12-camera samples or on custom synchronized MP4 datasets, including Kafka metadata and OSD/BEV output. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Optional] <br>
**Credential Type(s):** [Other [operator-managed MQTT/Kafka broker adapter credentials file]] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Setup And Readiness](references/setup.md) <br>
- [Sample Runs](references/sample-run.md) <br>
- [Custom Dataset Runs](references/custom-dataset.md) <br>
- [AMC Calibration Handoff](references/amc-calibration-handoff.md) <br>
- [Generate DeepStream Configs](references/generate-configs.md) <br>
- [Container Networking](references/container-network.md) <br>
- [Visualization And Metadata](references/visualization-metadata.md) <br>
- [Cleanup And Teardown](references/teardown.md) <br>
- [NVIDIA DeepStream repository](https://github.com/NVIDIA/DeepStream.git) <br>
- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Files, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks; generated DeepStream config files, MP4 videos, and Kafka protobuf metadata] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Run artifacts are written under the experiment directory and verified against the run start time; broker credential files are kept local with restricted permissions.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (4 positive), 1 attempt per task, each run in an isolated k8s sandbox pod and compared against a no-skill baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? <br>
- Correctness: Is the answer correct? <br>
- Discoverability: Was the right skill loaded when needed? <br>
- Effectiveness: Did the skill help complete the task? (equal-weight goal completion and expected workflow adherence) <br>
- Efficiency: Did it avoid wasted tool calls and token usage? (tool-call productivity and token efficiency) <br>

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
| Overall | 78.8% — baseline ran, but no comparable score was available; uplift unavailable | 81.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 87.5% → 100.0% (+12.5 points) |
| Correctness | 30.0% → 85.0% (+55.0 points) | 45.0% → 70.0% (+25.0 points) |
| Discoverability | 71.3% — baseline ran, but no comparable score was available; uplift unavailable | 91.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 38.4% → 56.3% (+17.9 points) | 43.4% → 59.9% (+16.5 points) |
| Efficiency | 81.4% — baseline ran, but no comparable score was available; uplift unavailable | 87.6% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
