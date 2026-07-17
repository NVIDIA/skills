## Description: <br>
Orchestration skill for NVIDIA Nemotron Speech (Riva) / NeMo ASR domain and language adaptation that scopes the task, picks the cheapest sufficient path (word boosting, n-gram LM, or fine-tuning), delegates each stage to the right sub-skill, and answers cost/time/data questions along the way. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers who need to improve or customize NVIDIA Nemotron Speech / Riva ASR for a specific domain or language, including reducing WER, adding a language, or planning a fine-tune pipeline. <br>

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
- [NVIDIA Speech NIM Documentation](https://docs.nvidia.com/nim/speech/latest/index.html) <br>
- [ASR Customization Guide](https://docs.nvidia.com/nim/speech/latest/asr/customization/customization.html) <br>
- [ASR Support Matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html) <br>
- [Riva ASR Tutorials](https://github.com/nvidia-riva/tutorials) <br>
- [Tokenizer Extension to New Language + Acoustic Fine-Tune](https://github.com/nvidia-riva/tutorials/blob/main/asr-extend-tokenizer-to-newlang-ft-acoustic-model.ipynb) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Configuration instructions, Shell commands] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`claude-code`) <br>
- Codex (`codex`) <br>



## Evaluation Tasks: <br>
Evaluated against 9 evaluation tasks (6 positive activation, 3 negative activation) with 1 attempt per task in the astra-sandbox environment. <br>

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
| Security | 8 | 100% (+0%) | 100% (+11%) |
| Correctness | 8 | 80% (+28%) | 88% (+27%) |
| Discoverability | 8 | 86% (+47%) | 93% (+50%) |
| Effectiveness | 8 | 64% (+20%) | 76% (+24%) |
| Efficiency | 8 | 83% (+36%) | 88% (+37%) |

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
