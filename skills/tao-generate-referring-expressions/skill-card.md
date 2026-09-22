## Description: <br>
Four-step image referring-expression pipeline that turns images plus KITTI bounding-box labels into region descriptions, scene captions, grounded referring expressions, and optionally verified expressions via VLM distillation. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers who need to generate referring-expression and grounding annotations from images with KITTI-format bounding-box labels using VLM-based auto-labeling. <br>

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
- [Full Configuration Reference](references/configuration.md) <br>
- [vLLM Server Setup](references/vllm_server.md) <br>
- [TAO Skill Bank Repository](https://github.com/NVIDIA-TAO/tao-skill-bank) <br>


## Skill Output: <br>
**Output Type(s):** [Files, Shell commands, Configuration instructions] <br>
**Output Format:** [JSONL annotation files with Markdown-formatted shell commands] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Per-step JSONL outputs (seed, region_expr, image_caption, grounding_expr, double_check) with optional legacy .txt format] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
1 evaluation task (1 positive) evaluated per agent with 3 attempts each in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether the skill is safe to use, covering unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Checks whether the final answer is correct against the reference answer. <br>
- Discoverability: Checks whether the right skill was loaded and activated when needed. <br>
- Effectiveness: Checks whether the skill helped complete the user's goal and expected workflow (equal-weight mean of goal completion and behavior adherence). <br>
- Efficiency: Checks whether the skill avoided wasted tool calls and token usage (50% tool-call productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Detects unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `accuracy`: Measures final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Measures whether the user's goal was achieved. <br>
- `behavior_check`: Measures whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Measures tool-call productivity (routing scored under Discoverability). <br>
- `token_efficiency`: Measures actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 99.6% | 74.7% |
| Security | 100.0% → 100.0% (±0.0 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 6.7% → 100.0% (+93.3 points) | 20.0% → 100.0% (+80.0 points) |
| Discoverability | 100.0% | 0.0% |
| Effectiveness | 16.7% → 100.0% (+83.3 points) | 48.3% → 75.0% (+26.7 points) |
| Efficiency | 97.7% | 99.7% → 98.6% (-1.1 points) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
