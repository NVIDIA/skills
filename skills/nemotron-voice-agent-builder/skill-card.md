## Description: <br>
Create, refine, or fix NVIDIA voice agents (Cascaded or Omni) with Pipecat or LiveKit, covering speech (ASR/TTS) customization and cloud or local deployment. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers building, scaffolding, configuring, refining, or fixing real-time NVIDIA voice agents using Cascaded (ASR → LLM → TTS) or Omni pipelines on Pipecat or LiveKit frameworks, with cloud or local deployment. <br>

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
- [Intake](references/intake.md) <br>
- [Output Contract](references/output-contract.md) <br>
- [Preflight](references/preflight.md) <br>
- [Agent Behavior](references/domain/agent-behavior.md) <br>
- [Speech Customization](references/domain/speech-customization.md) <br>
- [Pipecat Framework](references/frameworks/pipecat.md) <br>
- [LiveKit Framework](references/frameworks/livekit.md) <br>
- [Omni Pipeline](references/frameworks/omni.md) <br>
- [Model Catalog](references/models/catalog.md) <br>
- [ASR Models](references/models/asr.md) <br>
- [LLM Models](references/models/llm.md) <br>
- [TTS Models](references/models/tts.md) <br>
- [Language Routing](references/models/language-routing.md) <br>
- [Deployment](references/platforms/deployment.md) <br>
- [DGX Spark](references/platforms/dgx-spark.md) <br>
- [Jetson Thor](references/platforms/jetson-thor.md) <br>
- [Single GPU](references/platforms/single-gpu.md) <br>
- [Platform Readiness](references/platforms/readiness.md) <br>
- [Run Operations](references/operations/run.md) <br>
- [Iterate Operations](references/operations/iterate.md) <br>
- [Troubleshoot Operations](references/operations/troubleshoot.md) <br>
- [Observability](references/operations/observability.md) <br>
- [Remote WebRTC](references/networking/remote-webrtc.md) <br>
- [NVIDIA ASR Customization Docs](https://docs.nvidia.com/nim/speech/latest/asr/customization/customization.html) <br>
- [NVIDIA TTS Customization Docs](https://docs.nvidia.com/nim/speech/latest/tts/customization.html) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Files, Shell commands, Configuration instructions] <br>
**Output Format:** [Generated project files (Python, Docker Compose, shell scripts) with Markdown instructions] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
5 evaluation tasks (4 positive, 1 negative), 3 attempts per task, in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Checks whether the user's goal was achieved and the expected workflow behavior was followed (equal-weight mean of goal completion and behavior adherence). <br>
- Efficiency: Checks tool-call productivity and token efficiency (50% each). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Verifies absence of unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Verifies final-answer correctness against the reference answer. <br>
- `skill_execution`: Verifies the expected skill was selected and the workflow executed. <br>
- `goal_accuracy`: Verifies the user's goal was achieved. <br>
- `behavior_check`: Verifies the expected workflow behavior was followed. <br>
- `skill_efficiency`: Verifies tool-call productivity. <br>
- `token_efficiency`: Verifies actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | Not available | 87.1% |
| Security | Not available | 36.4% → 80.0% (+43.6 points) |
| Correctness | Not available | 54.6% → 100.0% (+45.4 points) |
| Discoverability | Not available | 90.0% |
| Effectiveness | Not available | 34.3% → 88.5% (+54.2 points) |
| Efficiency | Not available | 76.9% |

## Skill Version(s): <br>
2.2.0 (source: frontmatter, pyproject.toml, CHANGELOG) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
