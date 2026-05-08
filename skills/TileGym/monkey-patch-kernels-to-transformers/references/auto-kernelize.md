# Auto Kernelize
Autonomously create and integrate TileGym cuTile kernels to `transformers` model.

## Setup
Work with user to prepare experiment environment:
1. Check Git branch status. The previous commit should only contain monkey-patching existing TileGym OPs to the target transformers model. No other unstaged/uncommitted modifications
2. Check GPU available and UUID match; Docker container has been built
3. Study code relating to the target transformers model:
   - modeling/transformers/bench_<submodule_name>.sh: end-to-end benchmark entrance, run PyTorch baseline perf, cuTile kernelize perf, cuTile kernel coverage
   - src/tilegym/transformers/<submodule_name>/modeling_<submodule_name>.py: Target model specific OP adapters and wrappers
   - @src/tilegym/transformers/monkey_patch.py: Study apply_tilegym_kernel_to_<submodule_name> to understand how to kernelize
   - @modeling/transformers/infer.py: End-to-end benchmark and kernel coverage script
4. Create sandbox/<submodule_name>_results.md to track progress. The first run will write a baseline
5. Confirm and go: Once you get confirmation, kick off the experimentation

## Experimentation
Every experiment run on a NVIDIA GPU that [TileIR supported](https://docs.nvidia.com/cuda/tile-ir/latest/sections/stability.html#supported-architectures) (currently Ampere, Ada, and Blackwell architectures). Each experiment should be enforced to finish in 15 minutes. Every command should be executed within the experiment Docker container. `cd` to @modeling/transformers/ first, then `bash bench_<submodule_name>.sh` to launch one experiment.

### The goal
- Improve the **core metric**: cuTile kernel coverage percentage in terms of GPU time
- Subject to the **core constraint**: End-to-end throughput shall not drop compared to baseline

### What you can change
- @src/tilegym/transformers/<submodule_name>/modeling_<submodule_name>.py: New cuTile DSL kernels, wrappers, and other logic relating to model itself
- @src/tilegym/transformers/monkey_patch.py: Only change the `apply_tilegym_kernel_to_<submodule name>` function
- @modeling/transformers/infer.py: Only change
  * `apply_tilegym_kernel_to_<submodule name>` arguments
  * `KernelFilter.kernel_names_prefix` for new cuTile kernels
- @modeling/transformers/bench_<submodule_name>.sh: Optionally comment out line 30-38 to skip PyTorch throughput to accelerate experiment iterations. If so, ensure to restore on each experiment end
- @sandbox/: Feel free to add new files or modify files created by you, but don't check to git

### What you can NOT change
- Anything not listed above

### What to expect from experiment outputs
`bench_<submodule_name>.sh` prints ~300 lines of plain text. Use this command to grep core metrics: `grep -E "Average throughput|cuTile Kernel Coverage \(GPU Time\)" <output_file>`. Example output:

```text
Average throughput: 25.93 ± 3.20 tokens/sec
Average throughput: 53.41 ± 0.25 tokens/sec
>>> cuTile Kernel Coverage (GPU Time):    49.21% <<<
```

The first throughout corresponds to PyTorch baseline. The second cuTile.

### Track experiment progress
Use sandbox/<submodule_name>_results.md to record each experiment results. It should only contain a Markdown table with 5 columns:
- `commit`: git commit hash, 8 hexdigits
- `cuTile coverage`: greped cuTIle kernel coverage, two decimal point
- `cuTile throughput`: greped average value, no std, two decimal point
- `status`: Whether this experiment was `keep`, `discard`, `timeout`, or `crash`
- `description`: Concise text description of what was tried

Example content:

```markdown
| commit | cuTile coverage | cuTile throughput | status | description |
|:-------|----------------:|------------------:|:-------|:------------|
| 7241bf16 | 49.21 | 53.41 | keep | baseline |
```

Create the tabular header if the file was empty. Append one line for currently experiment.

### The baseline
The first experiment will not change any code and simply run bench_<submodule_name>.sh. Results will list at first row as baseline.

## The experiment loop
Core methodology is to create new cuTile kernels to replace uncovered PyTorch code while keeping performant and correctness. Try one piece of code at a time, and have clean experiment records.

LOOP:
1. Check git status: Current git branch/commit we're on
2. Identify one piece of uncovered PyTorch code and create cuTile kernels if it's straightforward; Otherwise delegate to a code subagent and let it follow /cutile-python SKILL
3. Integrate the new kernel to the transformers model and measure perf, coverage, and correctness (integrated model should produce meaningful results similar to baseline)
4. If crash at any previous step, or integrated model produced garbage outputs, try to fix. If you can't get things to work after more than a few attempts, give up
5. Git commit
6. Record results to sandbox/<submodule_name>_results.md
7. If coverage improved while throughput didn't drop and model output correct, you "advance" the branch, keeping the git commit
8. Otherwise, you git reset back to where you started

UNTIL: All target transformers model's PyTorch code was covered or user interrupted

*Be autonomous*: Ask user clarifications at setup phase. Once stepped into the experiment loop, do not pause to ask user feedback: Use your best judgement for decision making, search external resources and literatures promptly, and think harder if stuck.
