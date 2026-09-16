# DiffDock Parameter Guidance

Field types, required flags, and limits live in the canonical schema table in
[`api.md`](api.md) under **Request Body Schema**. This file is usage guidance
only.

## Ligand Format Choices

- Use SMILES plus `ligand_file_type="txt"` for simple single-ligand requests.
- Use SDF when the ligand has a known protonation state, stereochemistry, or 3D
  conformer that should be preserved.
- Use multi-entry SDF for batch docking when appropriate, and keep output file
  naming explicit because returned poses and confidences are parallel arrays.

## Practical Tuning

- Start with `num_poses=10`, `time_divisions=20`, and `steps=18`.
- Increase `num_poses` before changing diffusion controls if the issue is pose
  diversity.
- Keep `save_trajectory=false` unless the user explicitly asks for trajectory
  artifacts.
- For reproducible reporting, record all parameter values, receptor source,
  ligand source, endpoint mode, NIM image/version if local, and elapsed time.
