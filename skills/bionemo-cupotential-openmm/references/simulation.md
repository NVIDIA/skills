# Conversion, execution and restart

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Use the relevant section for force conversion, integration policy or restart.
The API notes below were checked against source 0.3.5. Verify version-specific
features in the installed plugin before applying them.

## Check before conversion

| Input property | Reviewed behavior / action |
|---|---|
| Standard nonbonded method | PME and LJPME can be converted. Default electrostatic splitting is Prolate. |
| Accuracy/mesh | The force policy is fixed at `5e-4`. Source meshes must be unpinned; a pinned alpha must match the derived value. Explain incompatible settings rather than clearing them silently. |
| Geometry | Orthorhombic and reduced triclinic cells are supported within stencil constraints. Each orthorhombic dimension must be at least three times the cutoff. Preserve the supplied box/cutoff. |
| Precision | The plugin's CUDA path supports single/mixed; a stock double-precision reference is a separate calculation. |
| Exceptions | Preserve pair definitions and the source exception-PBC flag; direct and reciprocal force-group separation is not preserved. |
| NPT | Context creation allocates a finite box envelope. Match the active barostat and supported box headroom; inspect envelope errors instead of changing pressure or box dimensions to hide them. |

If a user requires a currently unsupported model, report the specific boundary
and provide supported alternatives as choices. A switch to another ensemble,
cutoff model or force-field treatment is a user-visible scientific change.

## Replace a force once

This function accepts a supplied standard source System. It constructs every
replacement before deleting a source and leaves a stock copy available.

```python
import openmm as mm
import nv_openmm_plugin as nv

def converted_copy(stock_system):
    converted = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(stock_system))
    replacements = []
    for index, force in enumerate(converted.getForces()):
        if isinstance(force, mm.NonbondedForce):
            replacements.append((index, nv.NonbondedForce(force)))
    if not replacements:
        raise ValueError("No standard OpenMM NonbondedForce was selected")
    for index, replacement in reversed(replacements):
        converted.removeForce(index)
        converted.addForce(replacement)
    return converted
```

For a System with several nonbonded forces, confirm which ones the task requests
before using this function. Replacements are appended; if caller code stores
force indices, update that mapping. Preserve the stock System and compare
definitions of all untouched forces, masses, constraints and virtual sites.

## Integration and pressure policy

Force replacement does not require replacement of the stock integrator/barostat.
If selecting `nv.LangevinMiddleIntegrator`, inspect the source COM policy first:

```python
def transfer_com_policy(system):
    removers = [(index, force.getFrequency())
                for index, force in enumerate(system.getForces())
                if isinstance(force, mm.CMMotionRemover)]
    if len(removers) > 1:
        raise ValueError("Resolve multiple COM removers before conversion")
    frequency = removers[0][1] if removers else 0
    for index, _ in reversed(removers):
        system.removeForce(index)
    return frequency
```

Pass that frequency as `comFrequency` to the new integrator with the user's
temperature, friction and timestep. Retaining the stock integrator means retaining
its COM remover. Hydrogen mass repartitioning and constraints are part of the
input model, not an automatic prerequisite to copy from a 4 fs example.

For NPT, retain the selected stock barostat or replace the intended isotropic
barostat with `nv.MonteCarloBarostat`. Copy its pressure, temperature, frequency
and seed policy before removal. Do not add a second active barostat. Other pressure
coupling schemes require an explicit support check.

Complete force/integrator/barostat configuration before constructing a CUDA
Context. Use explicit supported precision. For an existing simulation, avoid
resetting time, step count or velocities while attaching reporters.

## CPU reference control

This control uses stock OpenMM only. It exercises the bundled synthetic model's
physical settings, finite energy/forces, step counts and a fresh-process binary
checkpoint restart. It never imports cuPotential or creates a CUDA Context.
Its stock integrator retains the explicit COM remover. Its NPT control uses
stock `MonteCarloBarostat`; native cuPotential policies still need GPU testing.

If stock OpenMM is absent and an isolated installation is permitted, use the
selected compatible interpreter to create a new environment, for example:

```bash
python3 -m venv /workspace/output/openmm-cpu-env
/workspace/output/openmm-cpu-env/bin/python -m pip install 'openmm==8.4.0.post2'
```

The base OpenMM package suffices; CUDA extras are unnecessary for this control.
Resolve the helper from the installed skill directory. The output must be a new
directory; preserve earlier results and their failures:

```bash
python -B /path/to/cupotential-openmm/scripts/run_cpu_reference.py run \
  --model /workspace/input/system.json \
  --output /workspace/output/nvt-cpu-reference \
  --ensemble nvt --steps 100 --resume-steps 40
```

For NPT use `--ensemble npt --resume-steps 50`. `--platform Reference` is an
explicit alternative to `CPU`. The helper writes input hashes, definitions,
force inventory, declared restoration criteria, raw states, checkpoint,
subprocess evidence and status. It compares physical state before continuing;
it does not certify identical future stochastic trajectories. A CPU pass leaves
cuPotential CUDA execution and candidate-versus-reference accuracy untested.
If CPU execution also fails, retain the exception and report that check as failed
or unavailable; prepared code alone is not a completed CPU run.

## Save and resume

Keep a compatible System definition, integrator definition and binary checkpoint
together with the runtime/OpenMM versions, precision and device identity:

```python
from pathlib import Path

def save_restart(context, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "system.xml").write_text(
        mm.XmlSerializer.serialize(context.getSystem()))
    (directory / "integrator.xml").write_text(
        mm.XmlSerializer.serialize(context.getIntegrator()))
    (directory / "state.chk").write_bytes(context.createCheckpoint())

def restore_restart(directory, precision="mixed"):
    directory = Path(directory)
    system = mm.XmlSerializer.deserialize((directory / "system.xml").read_text())
    integrator = mm.XmlSerializer.deserialize((directory / "integrator.xml").read_text())
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("CUDA"),
                         {"Precision": precision})
    context.loadCheckpoint((directory / "state.chk").read_bytes())
    return system, integrator, context
```

Import the plugin before deserializing plugin classes. Use a new output directory
for an independent run. Keep output append/overwrite behavior explicit on resume.
Binary checkpoints are tied to a compatible implementation/environment; an XML
State is a different portability tradeoff and does not restore all hidden state.

Test restoration in a fresh process. Compare positions, velocities, periodic box,
time, step count and live parameters with their saved values, then execute further
steps. NPT restoration does not promise replay of the future trajectory: OpenMM
barostat phase/adaptation bookkeeping is not fully checkpointed. Assess the
requested continuation guarantee separately from restored physical state.

Sources: [OpenMM component guide](https://docs.nvidia.com/cuda/cupotential/nonbonded-force),
the selected plugin's Python factory documentation, and its checkpoint tests.
For model-preserving comparisons, see Shirts et al.,
[SAMPL5 engine comparison](https://pmc.ncbi.nlm.nih.gov/articles/PMC5581938/).
