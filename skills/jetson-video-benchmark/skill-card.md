## Description: <br>
Use when measuring Jetson Video Codec SDK or PyNvVideoCodec encode/decode throughput, comparing presets or surfaces, testing codec-worker capacity with authenticated samples and user media, or producing a clearly labeled documentation-derived planning estimate when representative media is absent. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and engineers measuring Jetson Video Codec SDK encode/decode throughput, comparing codec presets or surfaces, testing codec-worker capacity, or obtaining documentation-derived planning estimates for Jetson platforms. <br>

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
- [Benchmark Output Contract](references/benchmark-output-contract.md) <br>
- [Benchmark Workflow](references/benchmark-workflow.md) <br>
- [Documented Performance Estimates](references/documented-performance-estimates.md) <br>
- [NVENC Application Note (Video Codec SDK 13.0)](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-application-note/index.html) <br>
- [NVDEC Application Note (Video Codec SDK 13.0)](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-application-note/index.html) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Files] <br>
**Output Format:** [JSON and Markdown] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
7 evaluation tasks (7 positive), 3 attempts per task, evaluated in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use — checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was selected when needed, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Whether the skill helped complete the user's goal (goal completion and expected workflow adherence). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage (tool-call productivity and token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (routing scored under Discoverability, not Efficiency). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | Not available | 94.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 82.1% → 100.0% (+17.9 points) |
| Correctness | Not available | 52.9% → 100.0% (+47.1 points) |
| Discoverability | Not available | 94.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 34.8% → 93.0% (+58.2 points) |
| Efficiency | Not available | 85.2% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
74bce4d (source: git SHA, committed 2026-09-16) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
