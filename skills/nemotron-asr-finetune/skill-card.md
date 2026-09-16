## Description: <br>
Orchestration skill for NVIDIA Nemotron Speech (Riva) / NeMo ASR domain and language adaptation that scopes the task, picks the cheapest sufficient path (word boosting, n-gram LM, or fine-tuning), delegates each stage to the right sub-skill, and answers cost/time/data questions along the way. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers who need to improve ASR accuracy for specific domains or languages using NVIDIA Nemotron Speech / Riva, including planning customization paths, orchestrating fine-tuning workflows, and estimating cost/time/data requirements. <br>

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
- [Orchestration Workflow](references/workflow.md) <br>
- [Path Selection Guide](references/path-selection.md) <br>
- [Planning Answers (Cost/Time/Data)](references/planning-answers.md) <br>
- [Sub-Skills Registry](references/sub-skills.md) <br>
- [NVIDIA NIM Speech ASR Customization Guide](https://docs.nvidia.com/nim/speech/latest/asr/customization/customization.html) <br>
- [NVIDIA NIM Speech Docs](https://docs.nvidia.com/nim/speech/latest/index.html) <br>
- [ASR Support Matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html) <br>
- [Riva ASR Tutorials](https://github.com/nvidia-riva/tutorials) <br>
- [Tokenizer Extension to New Language + Acoustic Fine-Tune](https://github.com/nvidia-riva/tutorials/blob/main/asr-extend-tokenizer-to-newlang-ft-acoustic-model.ipynb) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Configuration instructions, Shell commands] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
17 evaluation tasks (14 positive, 3 negative), 3 attempts per task in isolated k8s-sandbox pods. Evaluator version 1.5.6. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use — checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was selected when needed and decoys were avoided. <br>
- Effectiveness: Whether the skill helped complete the user's goal (50% goal completion + 50% expected workflow adherence). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage (50% tool-call productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity (routing scored under Discoverability). <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 90.7% | 80.4% |
| Security | 100.0% → 94.1% (-5.9 pts) | 85.2% → 80.0% (-5.2 pts) |
| Correctness | 61.0% → 94.1% (+33.1 pts) | 58.5% → 83.0% (+24.5 pts) |
| Discoverability | 98.2% | 91.8% |
| Effectiveness | 37.1% → 83.3% (+46.2 pts) | 38.1% → 66.7% (+28.6 pts) |
| Efficiency | 83.7% | 80.3% |

## Skill Version(s): <br>
1.3.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
