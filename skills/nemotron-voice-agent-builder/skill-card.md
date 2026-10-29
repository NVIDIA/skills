## Description: <br>
Create, refine, or fix NVIDIA voice agents (Cascaded or Omni) with Pipecat or LiveKit, including speech (ASR/TTS) customization and cloud or local deployment. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to build, scaffold, configure, refine, or fix real-time NVIDIA voice agents using Cascaded (ASR → LLM → TTS) or Omni (audio-in LLM + TTS) pipelines on Pipecat or LiveKit, including speech customization, multilingual routing, and cloud or local deployment (NIM, vLLM, NeMo-Speech.cpp) on workstations, DGX Spark, or Jetson Thor. <br>

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
- [Intake](references/intake.md) <br>
- [Preflight](references/preflight.md) <br>
- [Output Contract](references/output-contract.md) <br>
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
- [Platform Readiness](references/platforms/readiness.md) <br>
- [Deployment](references/platforms/deployment.md) <br>
- [Single GPU](references/platforms/single-gpu.md) <br>
- [DGX Spark](references/platforms/dgx-spark.md) <br>
- [Jetson Thor](references/platforms/jetson-thor.md) <br>
- [Run](references/operations/run.md) <br>
- [Iterate](references/operations/iterate.md) <br>
- [Troubleshoot](references/operations/troubleshoot.md) <br>
- [Observability](references/operations/observability.md) <br>
- [Remote WebRTC](references/networking/remote-webrtc.md) <br>
- [NVIDIA Speech NIM ASR Customization](https://docs.nvidia.com/nim/speech/latest/asr/customization/customization.html) <br>
- [NVIDIA Speech NIM TTS Customization](https://docs.nvidia.com/nim/speech/latest/tts/customization.html) <br>
- [Pipecat](https://github.com/pipecat-ai/pipecat) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Files, Shell commands, Configuration instructions] <br>
**Output Format:** [Generated project files (Python, Docker Compose, shell scripts, environment variable templates) with Markdown instructions] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Files are written only after user approval; handover requires a passing smoke test and a successful spoken exchange.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
5 evaluation tasks (4 positive, 1 negative), 1 attempt per task, each run in an isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal. <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal. <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% `skill_efficiency` and 50% `token_efficiency`. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 80.6% — baseline ran, but no comparable score was available; uplift unavailable | 79.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 40.0% → 80.0% (+40.0 points) | 50.0% → 90.0% (+40.0 points) |
| Correctness | 52.0% → 92.0% (+40.0 points) | 64.0% → 80.0% (+16.0 points) |
| Discoverability | 95.0% — baseline ran, but no comparable score was available; uplift unavailable | 93.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 34.0% → 46.0% (+12.0 points) | 28.0% → 50.0% (+22.0 points) |
| Efficiency | 89.9% — baseline ran, but no comparable score was available; uplift unavailable | 81.3% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
2.3.1 (source: frontmatter, pyproject.toml, CHANGELOG released 2026-10-29) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
