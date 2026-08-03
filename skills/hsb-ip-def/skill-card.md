## Description: <br>
Generate, validate, compare, or explain HSB HOLOLINK_def.svh macros for the NVIDIA Holoscan Sensor Bridge FPGA IP, using bundled Python scripts that produce validated .svh output files after user-confirmed paths. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and FPGA engineers generating, validating, comparing, or reasoning about HOLOLINK_def.svh configuration headers for NVIDIA Holoscan Sensor Bridge IP designs. <br>

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
- [Generate Workflow](references/generate-workflow.md) <br>
- [Script Usage](references/script-usage.md) <br>
- [Macro Reference](references/macro-reference.md) <br>
- [Validation Rules](references/validation-rules.md) <br>
- [Archetypes](references/archetypes.md) <br>
- [Init Reg Cookbook](references/init-reg-cookbook.md) <br>
- [Top Port Map](references/top-port-map.md) <br>
- [Advanced Macros](references/advanced-macros.md) <br>
- [HSB IP Integration Guide](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/blob/release-2.6.0-EA/docs/user_guide/ip_integration.md) <br>
- [HSB Port Description](https://github.com/nvidia-holoscan/holoscan-sensor-bridge/blob/release-2.6.0-EA/docs/user_guide/port_description.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Files, Shell commands, Analysis] <br>
**Output Format:** [SystemVerilog header files (.svh) and structured validation output (JSON/text)] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- `claude-code` <br>
- `codex` <br>



## Evaluation Tasks: <br>
Evaluated against 4 evaluation tasks in the `external` NVSkills-Eval profile within an astra-sandbox environment. <br>

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
| Security | 4 | 100% (+0%) | 75% (-25%) |
| Correctness | 4 | 97% (+64%) | 88% (+50%) |
| Discoverability | 4 | 95% (+59%) | 64% (+11%) |
| Effectiveness | 4 | 68% (+59%) | 74% (+63%) |
| Efficiency | 4 | 81% (+53%) | 55% (+10%) |

## Skill Version(s): <br>
0.1.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
