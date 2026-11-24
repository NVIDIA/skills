# CCCL skill evals

The cases are split into two datasets:

- `evals.json`: questions that need no CUDA toolkit or GPU. SkillEvaluator
  loads only this file, so it is the dataset NVSkills CI runs.
- `evals-gpu.json`: tasks that build and run CUDA code (porting, build fixes,
  performance work, writing kernels). They need a CUDA 13.4 toolkit and a GPU,
  which the CI sandbox does not provide, so CI does not run them.

To run the GPU cases on a machine with a GPU, copy `evals-gpu.json` over
`evals.json` in a scratch copy of the skill and run SkillEvaluator there. Both
datasets use the same `files/` directory.
