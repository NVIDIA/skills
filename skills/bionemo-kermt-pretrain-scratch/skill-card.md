## Description: <br>
Pretrain a fresh KERMT model from scratch on a user-provided corpus, building a new vocabulary from the corpus, instantiating the model architecture from defaults, and launching pretrain_ddp.py inside the kermt container. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers who need to pretrain a new KERMT molecular property prediction model from scratch on a custom chemistry corpus, rather than continuing from an existing checkpoint. <br>

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
- [KERMT paper — Multitask finetuning and acceleration of chemical pretrained models](https://arxiv.org/abs/2510.12719) <br>
- [GROVER paper — Self-Supervised Graph Transformer on Large-Scale Molecular Data](https://arxiv.org/abs/2007.02835) <br>
- [cuik-molmaker — GPU-accelerated molecular featurization](https://github.com/NVIDIA-Digital-Bio/cuik-molmaker) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 4 internal evaluation tasks (3 positive, 1 negative) with 3 attempts per task in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether the skill is safe to use, covering unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks whether the skill produces correct answers against the reference answer. <br>
- Discoverability: Checks whether the right skill was selected when needed, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal (goal_accuracy 50%) and followed expected workflow behavior (behavior_check 50%). <br>
- Efficiency: Checks whether the skill avoided wasted tool calls (skill_efficiency 50%) and token usage (token_efficiency 50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 79.9% — baseline ran, but no comparable score was available; uplift unavailable | 80.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 75.0% (-25.0 points) | 50.0% → 100.0% (+50.0 points) |
| Correctness | 6.7% → 90.0% (+83.3 points) | 91.4% → 90.0% (-1.4 points) |
| Discoverability | 93.3% — baseline ran, but no comparable score was available; uplift unavailable | 88.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 20.0% → 55.0% (+35.0 points) | 30.4% → 48.1% (+17.7 points) |
| Efficiency | 86.2% — baseline ran, but no comparable score was available; uplift unavailable | 76.0% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
77111e0 (source: git SHA, committed 2026-09-09) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
