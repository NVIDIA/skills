# DiffDock Validation

## Input Checks

- Confirm the receptor string contains at least one ATOM record.
- Strip non-ATOM lines from user-provided receptor PDB files unless the task
  explicitly requires retaining a supported record type.
- Confirm `ligand_file_type` matches the ligand representation:
  - `"txt"` for SMILES;
  - `"sdf"` for SDF;
  - `"mol2"` for MOL2.
- Reject empty ligand strings before calling the endpoint.
- For SMILES, optionally validate with RDKit when available, but do not make
  RDKit a hard requirement for the skill.

## Response Checks

- `ligand_positions` exists, is non-empty, and contains SDF-format strings.
- `position_confidence` exists and has the same length as `ligand_positions`.
- Confidence values are numeric and finite.
- Save each pose to a separate `.sdf` file with rank and confidence in the
  filename.
- Save request metadata and response metadata for reproducibility.

## Scientific Checks

- Treat the top-ranked pose as a hypothesis, not proof of binding.
- Inspect poses in PyMOL, ChimeraX, or a similar viewer with the receptor.
- Look for obvious clashes, disconnected fragments, pose outside plausible
  pockets, and missing expected interactions.
- Do not convert confidence directly into binding affinity.
- For decision-making, combine docking with orthogonal evidence such as
  experimental data, affinity prediction, physics-based refinement, or medicinal
  chemistry review.

## Local Deployment Checks

- Docker startup uses NGC auth and `LOCAL_NIM_CACHE`.
- Local readiness endpoint is `http://localhost:8000/v1/health/ready`.
- Local inference endpoint is
  `http://localhost:8000/molecular-docking/diffdock/generate` with no `/v1/`.
- After the container is healthy, local inference requests use no Authorization
  header.
