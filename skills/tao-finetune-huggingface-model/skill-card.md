## Description: <br>
Fine-tune any HuggingFace CV / VLM / LLM model on local NVIDIA GPUs inside an NGC PyTorch container when no dedicated TAO model skill matches. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers who need to fine-tune HuggingFace computer-vision, vision-language, or large-language models (full or LoRA) on local NVIDIA GPUs using NGC PyTorch containers when no dedicated TAO model skill covers the target model. The skill researches the recipe, generates and smoke-tests a reproducible training project, trains, evaluates, runs inference, optionally pushes to the HF Hub, and emits a rerun skill. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Optional] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [NVIDIA TAO Skill Bank](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>
- [NVIDIA Deep Learning Frameworks Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) <br>
- [Core Rules](references/core-rules.md) <br>
- [Detailed Workflow](references/detailed-workflow.md) <br>
- [Workflow: Intake and Preflight](references/workflow-intake-preflight.md) <br>
- [Workflow: Generate and Train](references/workflow-generate-train.md) <br>
- [Workflow: Push and Rerun](references/workflow-push-rerun.md) <br>
- [Docker Runs](references/docker-runs.md) <br>
- [Hardware and Container Selection](references/hardware-container.md) <br>
- [Research Priorities](references/research-priorities.md) <br>
- [Compatibility Workarounds](references/compat-workarounds.md) <br>
- [Error Playbook](references/error-playbook.md) <br>
- [CV Scripts](references/cv-scripts.md) <br>
- [VLM Scripts](references/vlm-scripts.md) <br>
- [Hub Push](references/hub-push.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Code, Files] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (4 positive), 1 attempt per task, each attempt in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the right skill was loaded when needed. <br>
- Effectiveness: Checks goal completion and expected workflow adherence. <br>
- Efficiency: Checks tool-call productivity and token usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 82.5% | 57.6% |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 30.0% → 100.0% (+70.0 points) | 35.0% → 70.0% (+35.0 points) |
| Discoverability | 57.5% | 0.0% |
| Effectiveness | 23.5% → 87.9% (+64.4 points) | 35.0% → 60.4% (+25.4 points) |
| Efficiency | 67.0% | Not available |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
