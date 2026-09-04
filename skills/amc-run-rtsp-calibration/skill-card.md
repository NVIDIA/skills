## Description: <br>
Calibrate a new dataset from live RTSP camera streams via the AutoMagicCalib REST API. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to calibrate camera systems from live RTSP streams, leveraging the VIOS recording service and AutoMagicCalib REST API to create calibration datasets without pre-recorded video files. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [No] <br>
**Credential Type(s):** [None] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [run_rtsp_calibration.py](scripts/run_rtsp_calibration.py) <br>
- [AMC Setup Calibration Stack Skill](../amc-setup-calibration-stack/SKILL.md) <br>
- [AMC Run Video Calibration Skill](../amc-run-video-calibration/SKILL.md) <br>
- [AMC Run Sample Calibration Skill](../amc-run-sample-calibration/SKILL.md) <br>


## Skill Output: <br>
**Output Type(s):** [API Calls, Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
6 evaluation tasks (5 positive, 1 negative) in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Verifies final-answer correctness against the reference answer. <br>
- Discoverability: Checks whether the expected skill was found and executed when needed. <br>
- Effectiveness: Measures goal completion and expected workflow adherence. <br>
- Efficiency: Evaluates routing quality, workspace-aware skill reads, and productive tool use. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Detects unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies whether the expected skill was found and executed. <br>
- `skill_efficiency`: Assesses routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Scores final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Measures whether the user's goal was achieved. <br>
- `behavior_check`: Checks whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 50% → 92% (+42 points) | 41% → 80% (+39 points) |
| Security | 100% → 100% (±0 points) | 33% → 83% (+50 points) |
| Correctness | 17% → 100% (+83 points) | 17% → 70% (+53 points) |
| Discoverability | 47% → 93% (+46 points) | 46% → 88% (+42 points) |
| Effectiveness | 40% → 82% (+41 points) | 54% → 64% (+10 points) |
| Efficiency | 45% → 87% (+42 points) | 53% → 94% (+41 points) |

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
