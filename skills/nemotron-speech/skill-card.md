## Description: <br>
Routes NVIDIA Nemotron Speech (Riva) NIM tasks — deploys, runs, and tests ASR, TTS, and NMT NIMs on build.nvidia.com or self-hosted. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and engineers deploying, testing, and operating NVIDIA Nemotron Speech (Riva) NIMs for speech-to-text, text-to-speech, and translation workflows on cloud-hosted or self-hosted infrastructure. <br>

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
- [ASR Support Matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html) <br>
- [TTS Support Matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/tts.html) <br>
- [NMT Support Matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/nmt.html) <br>
- [Prerequisites (Driver / GPU / OS)](https://docs.nvidia.com/nim/speech/latest/get-started/prerequisites.html) <br>
- [ASR Pipeline Configuration](https://docs.nvidia.com/nim/speech/latest/asr/customization/pipeline-configuration.html) <br>
- [TTS Custom Deployment](https://docs.nvidia.com/nim/speech/latest/tts/custom-deployment.html) <br>
- [TTS Customization](https://docs.nvidia.com/nim/speech/latest/tts/customization.html) <br>
- [TTS Voices and Emotional Styles](https://docs.nvidia.com/nim/speech/latest/tts/voices.html) <br>
- [TTS Zero-Shot Voice Cloning](https://docs.nvidia.com/nim/speech/latest/tts/voice-cloning.html) <br>
- [NGC Model Catalog](https://catalog.ngc.nvidia.com/models) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Code] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
18 evaluation tasks (14 positive, 4 negative) run locally with 1 attempt per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill was found and executed when needed. <br>
- Effectiveness: Whether the skill helps the agent complete the user's goal and follow the expected workflow. <br>
- Efficiency: Whether the skill avoids wasted tool or skill usage. <br>

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
| Overall | 69% → 92% (+23 points) | 63% → 84% (+22 points) |
| Security | 94% → 89% (-6 points) | 72% → 81% (+8 points) |
| Correctness | 87% → 96% (+9 points) | 83% → 92% (+9 points) |
| Discoverability | 54% → 100% (+45 points) | 58% → 89% (+32 points) |
| Effectiveness | 60% → 88% (+28 points) | 68% → 80% (+11 points) |
| Efficiency | 48% → 88% (+39 points) | 32% → 79% (+48 points) |

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
