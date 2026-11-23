# Reference selection, metrics and saved snapshots

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Read the section appropriate to model equivalence, static numerical accuracy,
or trajectory validity. The API context is source 0.3.5; numeric acceptance limits
must be justified for the actual task and runtime.

## Select a comparison that answers the question

| Question | Evidence |
|---|---|
| Was the requested model preserved? | Inventory particle/exception/bond definitions, constraints, masses, virtual sites, units, cutoff/switch/tail conventions and ensemble policy. |
| Are energy/forces numerically accurate? | Evaluate fixed common configurations against an independently constructed, sufficiently converged reference; record its residual error and precision controls. |
| Is an implementation internally consistent? | Analytic small-system checks, force/energy finite differences, symmetry, exclusions and live-update controls. |
| Does a trajectory support a physical conclusion? | Appropriate integration/ensemble tests and adequate equilibrated sampling of the requested observable. |

For periodic electrostatics, stock PME and default Prolate target the same
interaction through different approximations. Measure both against a common
reference when judging accuracy. Refine the stock reference's reciprocal mesh
and direct-space convergence, documenting the splitting parameter and boundary
conventions. Verify that further refinement changes the answer by less than the
reference budget. Do not enlarge the physical box to make a larger cutoff fit;
if geometry prevents a simple refinement, document the limitation and choose an
appropriate alternative reference. The plugin's own pinned-mesh restrictions
remain in force; reference settings belong to the separate reference calculation.

For truncated LJ, keep the physical cutoff, switch and tail definition fixed.
Increasing this cutoff generally changes the model. For LJPME/dispersion compare
the intended periodic sum and its energy/correction conventions separately.
Double precision reduces arithmetic error but does not remove truncation or
model-definition error. A constant energy offset and a parameter-dependent energy
change also have different implications; quantify the convention rather than
discarding a discrepancy automatically.

Use multiple relevant configurations and a small analytic control. Include
ordinary and exceptional pairs and, when claimed, tilted boxes and parameter
updates. Inspect both global RMS/L2 error and outliers; a global average can hide
a serious local error. Preserve user-specified limits even if another metric is
more favorable. An independent protocol calibration must not overwrite failures
from the original protocol.

## Canonical snapshot files

The helper consumes one candidate and one reference JSON object, each containing:

```json
{
  "positions_nm": [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
  "box_vectors_nm": [[3.6, 0.0, 0.0], [0.0, 3.6, 0.0], [0.0, 0.0, 3.6]],
  "potential_energy_kj_mol": -0.2,
  "forces_kj_mol_nm": [[0.3, 0.0, 0.0], [-0.3, 0.0, 0.0]]
}
```

This is a synthetic format example, not a measured cuPotential result. Set both
Contexts from the same canonical geometry before evaluation and save that input
geometry with each result. Record the actual build and force-model definitions
alongside the files; geometry equality does not itself establish model equality.

For an existing OpenMM Context, this function applies a supplied geometry and
captures a result without integration:

```python
import json
from pathlib import Path
import openmm as mm
from openmm import unit

def save_snapshot(context, geometry, destination):
    context.setPeriodicBoxVectors(*(
        mm.Vec3(*vector) * unit.nanometer for vector in geometry["box_vectors_nm"]))
    context.setPositions(unit.Quantity(geometry["positions_nm"], unit.nanometer))
    state = context.getState(getEnergy=True, getForces=True)
    payload = {
        "positions_nm": geometry["positions_nm"],
        "box_vectors_nm": geometry["box_vectors_nm"],
        "potential_energy_kj_mol": state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole),
        "forces_kj_mol_nm": state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer).tolist(),
    }
    Path(destination).write_text(json.dumps(payload, allow_nan=False, indent=2))
```

Run the comparator from its installed skill location. Without thresholds it
reports diagnostics and does not declare a pass. Budget flags are selected for
the task; use `--help` for the available checks. Energy uses
`absolute_difference <= energy_atol + energy_rtol * abs(reference_energy)`.
Force RMS is over particles of the vector error; maximum error is over Cartesian
components; relative L2 divides the full error norm by the reference force norm.
Requested relative-force checks are inconclusive for a zero reference norm.
Malformed, nonfinite, mismatched-geometry or otherwise unevaluable inputs are
errors, not passing comparisons. Exit codes: 0 diagnostics/pass, 1 failed checks,
2 invalid input or an unevaluable requested check.

## Dynamics and statistical interpretation

Use a deliberate NVE diagnostic with suitable constraints/integration to examine
energy behavior and timestep dependence. Total energy conservation is not an
acceptance rule for a thermostatted/barostatted production trajectory. For NVT/NPT,
use appropriate kinetic/ensemble diagnostics, accounting for constrained degrees
of freedom and COM policy. Exact trajectory equality is generally a different
claim from statistical agreement.

Choose equilibration and effective sample-size analysis for the requested
observable. Longer output alone does not establish convergence. A short smoke
test should be described with its actual duration and observed checks.

Scientific basis:

- Shirts et al. (2017), [cross-engine model comparisons](https://pmc.ncbi.nlm.nih.gov/articles/PMC5581938/).
- Deserno and Holm (1998), [particle-mesh accuracy](https://arxiv.org/abs/cond-mat/9807099).
- Liang et al. (2026), [Ewald summation with prolates](https://www.nature.com/articles/s41467-026-73232-8). The paper informs error analysis; it does not define this plugin's public controls.
- Merz and Shirts (2018), [physical validation tests](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0202764).
- Chodera (2016), [equilibration detection](https://pmc.ncbi.nlm.nih.gov/articles/PMC4945107/).
