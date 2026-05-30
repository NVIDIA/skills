## Description: <br>
Read and write large cuPyNumeric arrays to HDF5 with Legate's parallel, distributed HDF5 I/O (legate.io.hdf5: to_file, from_file, from_file_batched). <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 OR Apache-2.0 <br>
## Use Case: <br>
Developers and engineers who need to save or load cuPyNumeric arrays to and from HDF5 files for large-scale distributed HPC and scientific computing workflows. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Legate I/O API Documentation](https://docs.nvidia.com/legate/latest/api/python/io/index.html) <br>
- [cuPyNumeric GitHub](https://github.com/nv-legate/cupynumeric) <br>
- [HDF5 — The HDF Group](https://www.hdfgroup.org/solutions/hdf5/) <br>
- [VFD GDS Plugin](https://github.com/nv-legate/vfd-gds) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions] <br>
**Output Format:** [Markdown with inline Python code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`claude-code`) <br>
- Codex (`codex`) <br>



## Evaluation Tasks: <br>
Evaluated against 17 tasks (11 positive activation, 6 negative activation) with 2 attempts per task via NVSkills-Eval. <br>

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
| Security | 8 | 100% (+6%) | 100% (+0%) |
| Correctness | 8 | 90% (+4%) | 93% (+9%) |
| Discoverability | 8 | 80% (+17%) | 80% (+7%) |
| Effectiveness | 8 | 90% (+4%) | 92% (+16%) |
| Efficiency | 8 | 80% (+24%) | 73% (+7%) |

## Skill Version(s): <br>
2.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
