# Environment selection and execution probe

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Use this reference when choosing an installation route or investigating a failed
import/Context. The reviewed contract is cuPotential 0.3.5; resolve matching
release documentation for a different installation.

## Routes and compatibility

For the reviewed release, prebuilt OpenMM plugin packages target Linux x86-64,
Python 3.12, OpenMM 8.4 with CUDA support, and supported NVIDIA GPUs. The source
documents CUDA 13.4 and SM 75/80/86/89/90/100/103/120. Driver R615 or newer is the
direct compatibility route. Supported data-center/professional GPUs can also use
the documented CUDA forward-compatibility route on selected older driver
branches. Check the current installation and NVIDIA compatibility guides for the
particular GPU/driver combination; a container does not supply a host GPU driver.

For a compatible consumer environment:

```bash
python -m pip install nv-openmm-plugin
```

Pin the requested verified release when reproducibility requires it. For an
existing environment, inspect its constraints before changing OpenMM or CUDA
dependencies. A successful pip resolver is not an execution test.

For the source route, use a separate checkout of the selected release. For a
fresh checkout of the reviewed release, start in the directory where you want
the library source to live:

```bash
git clone --branch v0.3.5 https://github.com/NVIDIA/cuPotential.git
cd cuPotential
./run_docker bash
```

If a matching checkout already exists, change to its root and run
`./run_docker bash` there. The helper and Dockerfile belong to the upstream
cuPotential checkout. Its Docker workflow provides the compiler and dependencies.
Keep pip installation and verification in that same session: the helper creates
disposable containers, so changes confined to a container's filesystem are
discarded on exit.
Resolve release filenames from the actual release; source docs containing
`@RELEASE_VERSION@` are templates. Follow the matching release's build workflow
instead of patching a build helper from an older blog snippet.

## Read-only inventory

Run the bundled inspector with the target interpreter:

```bash
python /path/to/cupotential-install/scripts/inspect_environment.py
```

It records metadata and module discovery without importing the GPU libraries.
No distribution metadata can be normal for an SDK/source path. Check the actual
import next; distinguish its failure from missing wheel metadata.

## Minimal CUDA execution check

Run this separately in the selected environment. It creates a small neutral
periodic system and requests energy and forces. It establishes basic execution,
not production accuracy, speed or scientific suitability.

```python
import math
import openmm as mm
from openmm import unit
import nv_openmm_plugin as nv

system = mm.System()
stock = mm.NonbondedForce()
stock.setNonbondedMethod(mm.NonbondedForce.PME)
stock.setCutoffDistance(0.9 * unit.nanometer)
stock.setEwaldErrorTolerance(5e-4)
for charge in (0.1, -0.1, 0.1, -0.1):
    system.addParticle(39.9 * unit.dalton)
    stock.addParticle(charge, 0.30, 0.20)
system.setDefaultPeriodicBoxVectors(
    mm.Vec3(3.6, 0, 0), mm.Vec3(0, 3.6, 0), mm.Vec3(0, 0, 3.6))
system.addForce(nv.NonbondedForce(stock))
integrator = mm.VerletIntegrator(0.001 * unit.picosecond)
context = mm.Context(system, integrator,
                     mm.Platform.getPlatformByName("CUDA"), {"Precision": "mixed"})
context.setPositions(unit.Quantity(
    [(0.4, 0.4, 0.4), (0.9, 0.4, 0.4),
     (0.4, 1.0, 0.4), (0.9, 1.0, 0.4)], unit.nanometer))
state = context.getState(getEnergy=True, getForces=True)
energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
forces = state.getForces().value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
assert math.isfinite(energy)
assert all(math.isfinite(component) for force in forces for component in force)
assert any(abs(component) > 0 for force in forces for component in force)
print("CUDA energy/force evaluation completed:", energy, "kJ/mol")
```

If this fails, record `mm.Platform.getPluginLoadFailures()` when OpenMM imports,
the exception, imported module paths and selected environment. Inspect the
OpenMM CUDA plugin, cuPotential runtime, cuFFT/NVRTC/nvJitLink dependency set and
driver as indicated by the error. Avoid assuming that a library visible in a
different environment is loaded by this interpreter.

Sources: [installation](https://docs.nvidia.com/cuda/cupotential/install),
[CUDA forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html),
and the selected release's `Dockerfile`, `.cuda-version` and Python package metadata.
