## Description: <br>
Plan Nemotron customization pipelines from repo steps: SFT, PEFT/LoRA, AutoModel vs Megatron-Bridge, DPO/RLVR/GRPO/RLHF, curate-then-translate, BYOB/MCQ benchmark prep or translation, checkpoint conversion, ModelOpt optimization, and endpoint or checkpoint evaluation. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to plan and configure Nemotron model customization pipelines, including SFT, PEFT/LoRA, alignment (DPO/RLVR/GRPO/RLHF), data curation, translation, BYOB/MCQ benchmark preparation, checkpoint conversion, ModelOpt optimization, and endpoint or checkpoint evaluation. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [WORKFLOW.md](references/WORKFLOW.md) <br>
- [PROJECT.md](references/act/PROJECT.md) <br>
- [STAGE.md](references/act/STAGE.md) <br>
- [Context Index](references/context/index.toml) <br>
- [Nemotron Developer Docs](https://nvidia-nemo.github.io/Nemotron/dev/) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- `claude-code` <br>
- `codex` <br>



## Evaluation Tasks: <br>
Evaluated against 8 evaluation tasks from NVSkills-Eval (external profile, local environment, 2 attempts per task, 50% pass threshold). Overall verdict: PASS. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

Underlying evaluation signals used in this run: <br>
- `skill_execution`: Verifies that the agent loaded the expected skill and workflow. <br>
- `skill_efficiency`: Checks routing quality, decoy avoidance, and redundant tool usage. <br>
- `accuracy`: Grades final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the overall user task completed successfully. <br>
- `behavior_check`: Verifies expected behavior steps, including safety expectations. <br>
- `token_efficiency`: Compares token usage with and without the skill. <br>



## Evaluation Results: <br>
| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | 8 | 79% (-6%) | 79% (+10%) |
| Correctness | 8 | 95% (+12%) | 91% (+33%) |
| Discoverability | 8 | 74% (+35%) | 68% (+27%) |
| Effectiveness | 8 | 90% (+2%) | 86% (+32%) |
| Efficiency | 8 | 58% (+31%) | 55% (+16%) |

## Skill Version(s): <br>
0.1.1 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
