## Description: <br>
How to swap the VLM in the VSS Alerts Blueprint — covers RTVI-VLM microservice deployment methods, all three VLM consumers (rtvi-vlm, vlm-as-verifier, vss-agent), and health checks. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and engineers deploying or reconfiguring the VLM endpoint in the NVIDIA VSS Alerts Blueprint, including switching between OpenAI-compatible and in-container vLLM modes across the three independent VLM consumers. <br>

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
- [Health Checks](references/health-checks.md) <br>
- [Port and URL Wiring](references/port-and-url-wiring.md) <br>
- [VSS Source Layout](references/vss-source-layout.md) <br>
- [VSS Blueprint Repository](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) <br>
- [RT-VLM README (v3.2.1)](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/blob/v3.2.1/services/rtvi/rt-vlm/README.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
6 evaluation tasks (6 positive), each executed in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helped complete the user’s goal and followed the expected workflow (equal-weight mean of goal completion and behavior adherence). <br>
- Efficiency: Whether the skill avoided wasted tool or skill usage through quality routing and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user’s goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 54% → 95% (+42 points) | 54% → 84% (+29 points) |
| Security | 100% → 100% (±0 points) | 83% → 75% (-8 points) |
| Correctness | 43% → 93% (+50 points) | 47% → 83% (+37 points) |
| Discoverability | 50% → 100% (+50 points) | 49% → 86% (+38 points) |
| Effectiveness | 40% → 84% (+45 points) | 44% → 81% (+37 points) |
| Efficiency | 35% → 99% (+63 points) | 48% → 92% (+44 points) |

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
