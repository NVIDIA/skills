# Expression construction and numerical checks

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Use the recipe matching a standalone custom potential, replacement of standard
LJ, or an alchemical application. Reviewed source: cuPotential 0.3.5.

## Standalone periodic WCA example

The following builds the force definition for two particles. Substitute the
user's particle table, cutoff and globals when adapting it; attach one version
to each otherwise equivalent System before constructing independent Contexts.
The cutoff must cover the largest pair-specific WCA radius in the actual system.

```python
import openmm as mm
import nv_openmm_plugin as nv

def wca_definition():
    force = mm.CustomNonbondedForce(
        "strength*select(step(rc-r),4*eps*(s6*s6-s6)+eps,0);"
        "s6=(sig/r)^6;rc=2^(1/6)*sig;"
        "sig=(sigma1+sigma2)/2;eps=sqrt(epsilon1*epsilon2)")
    force.addPerParticleParameter("sigma")
    force.addPerParticleParameter("epsilon")
    force.addGlobalParameter("strength", 1.0)
    force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(0.9)
    force.setUseLongRangeCorrection(False)
    force.addParticle([0.30, 0.50])
    force.addParticle([0.35, 0.80])
    return force

stock_force = wca_definition()
plugin_force = nv.NonbondedForce(wca_definition())
```

Here sigma and distance are in nm; epsilon and the returned energy are in kJ/mol.
At positive separations inside the pair radius, verify against the independent
WCA formula. Outside it, energy and force vanish. At zero separation this formula
is singular: prepare valid coordinates rather than hiding an overlap.

After Context creation, `context.setParameter("strength", value)` changes the
global value. Compare values 1, a nontrivial intermediate value and the original
value at fixed coordinates. Include an excluded-pair control and nonzero forces.

### Independent analytic and finite-difference controls

Use a separately coded formula for the two-particle control. This returns energy
and the outward radial force magnitude, using nm and kJ/mol:

```python
import math

def analytic_wca(r, sigma_i, sigma_j, epsilon_i, epsilon_j, strength=1.0):
    if r <= 0:
        raise ValueError("The WCA control requires positive separation")
    sigma = (sigma_i + sigma_j) / 2
    epsilon = math.sqrt(epsilon_i * epsilon_j)
    if r >= 2**(1/6) * sigma:
        return 0.0, 0.0
    q6 = (sigma / r)**6
    energy = strength * epsilon * (4 * (q6*q6 - q6) + 1)
    radial_force = strength * 24 * epsilon / r * (2*q6*q6 - q6)
    return energy, radial_force
```

For displacement `d = x_i - x_j` and separation `r`, the Cartesian force is
`F_i = radial_force * d / r`, with `F_j = -F_i`. In particular, a repulsive pair
on the x axis with `x_i < x_j` has a negative x force on particle i.

Evaluate the candidate energy at `x_i+h` and `x_i-h` for each tested Cartesian
component; compare `-(U_plus-U_minus)/(2*h)` to the candidate force. Declare the
step sequence and budgets first. Choose a smooth interior configuration and
steps that stay clear of zero separation, the pair support boundary and periodic
image changes. Save the measured energy, analytic force, candidate force and
finite-difference force for each step. A nonzero interior test, an outside-support
test, an excluded-pair test and a live-global restoration test answer different
questions; retain each result separately.

## Replacing the cutoff part of a standard force

`nv.NonbondedForce(stock_nonbonded)` copies the standard force's charges,
exceptions and supported offsets. `setCutoffInteraction(custom_definition)`
replaces ordinary LJ with the complete supplied cutoff expression while retaining
the standard electrostatic terms and independent exceptions. Include ordinary
LJ explicitly if the requested custom expression is only an additional term.

### Complete LJ-plus-Gaussian construction

This pattern covers fixed particle parameters, standard PME, and disabled
switching/tails. It returns two separate custom definitions: Gaussian-only for
the stock reference, and the complete LJ-plus-Gaussian replacement for cuPotential.
It rejects offsets and policy changes that would need a different parameter law
or reference. `amplitude` is in kJ/mol and `width` is in nm.

```python
import math
import openmm as mm
from openmm import unit

def gaussian_definitions(source, amplitude, width):
    if not math.isfinite(amplitude) or not math.isfinite(width) or width <= 0:
        raise ValueError("Require finite amplitude and positive finite width")
    if source.getNonbondedMethod() != mm.NonbondedForce.PME:
        raise ValueError("This recipe expects standard PME")
    if (source.getNumParticleParameterOffsets() or
            source.getNumExceptionParameterOffsets()):
        raise ValueError("Offsets require an explicit matching parameter law")
    if source.getUseSwitchingFunction() or source.getUseDispersionCorrection():
        raise ValueError("This recipe expects switching and tails already disabled")

    addition = mm.CustomNonbondedForce("amplitude*exp(-0.5*(r/width)^2)")
    replacement = mm.CustomNonbondedForce(
        "4*eps*(q6*q6-q6)+amplitude*exp(-0.5*(r/width)^2);"
        "q6=(sig/r)^6;sig=(sigma1+sigma2)/2;"
        "eps=sqrt(epsilon1*epsilon2)")
    replacement.addPerParticleParameter("sigma")
    replacement.addPerParticleParameter("epsilon")
    for force in (addition, replacement):
        force.addGlobalParameter("amplitude", amplitude)
        force.addGlobalParameter("width", width)
        force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
        force.setCutoffDistance(source.getCutoffDistance())
        force.setUseSwitchingFunction(False)
        force.setUseLongRangeCorrection(False)
    for index in range(source.getNumParticles()):
        _, sigma, epsilon = source.getParticleParameters(index)
        addition.addParticle([])
        replacement.addParticle([
            sigma.value_in_unit(unit.nanometer),
            epsilon.value_in_unit(unit.kilojoule_per_mole)])
    for index in range(source.getNumExceptions()):
        first, second, *_ = source.getExceptionParameters(index)
        addition.addExclusion(first, second)
        replacement.addExclusion(first, second)
    return addition, replacement
```

Construct the candidate while its source force is still valid:

```python
import nv_openmm_plugin as nv

addition, replacement = gaussian_definitions(source, amplitude=0.1, width=0.3)
candidate = nv.NonbondedForce(source)
candidate.setCutoffInteraction(replacement)
```

Put `candidate` in the candidate System. Build the independent stock System from
a separate copy of `source` plus `addition`; do not also add `replacement` there.
Each force object belongs to one System. Preserve the source's exception laws and
other forces. Match the supplied coordinates, box, integrator and precision in
the two Contexts, then request and save energies and forces from both. The example
numbers are placeholders for the user's model, rather than validation budgets.
Retain observed approximation differences and failing comparisons.

Check particle count, cutoff, exclusions, switching and global-name compatibility
before fusing definitions. Interaction groups are not generally supported by the
custom conversion constructor. The supplied alchemical example lowers groups to
membership parameters; adapting that approach requires checking pair multiplicity
and tail conventions for the actual input.

Make switching and tail choices explicit. For a switched potential `S(r) U(r)`,
the derivative is `S U' + S' U`; a switch written into the expression and an
automatic switch would be applied twice. The default replacement removes the
native LJ tail. Preserving it and integrating a custom homogeneous tail are
different policies. Grouped and ungrouped averaging can also change energies.

`LatticeSumInteraction` declares supported periodic P1/P6 terms already present
in the full expression, with matching reciprocal/self/exclusion treatment. It
does not add a missing bare interaction. In the reviewed interface, LJPME/P6,
automatic switching and homogeneous tails have compatibility restrictions; check
the selected release instead of composing them by analogy.

## Force checks

At fixed model parameters compare the force with `-dU/dr`, and for multi-particle
cases compare Cartesian components with `-partial U/partial x`. Sweep displacement
size to find a stable finite-difference range. Single-precision subtraction makes
arbitrarily small steps unreliable. Keep cutoff-boundary tests separate from
smooth-interior finite differences. Declare required acceptance criteria before
the comparisons and preserve every failing required result.

Parameter values and expression/schema changes have different update paths.
Use the documented `updateParametersInContext` for supported particle-value
edits. Expression layout, excluded topology, lattice definitions and policy
changes can require Context reinitialization. Review the current method contract
for the particular change.

## Alchemical scope

Preserve the input lambda definitions, offsets, exceptional bonds, atom mapping
and endpoint physics. Live globals do not imply availability of energy derivatives
with respect to those globals; the reviewed standard offset path does not compute
them. Verify the requirements of a requested TI/estimation protocol explicitly.

For a contract-only assessment, distinguish live global updates, coordinate
forces, restart/checkpoint support and analytic `dU/dlambda`: none establishes
the others. If the supplied path lacks the analytic derivative required by the
estimator, record that incompatibility for that path. Choose a path whose
contract provides the required derivative, or explicitly change the estimator
and protocol. For example, BAR/MBAR would require cross-state energy evaluations
and appropriate ensembles; verify those capabilities before proposing execution.
Finite-difference energy estimates do not meet a requirement for an analytic
derivative. Save the decision and evidence needs even when no simulation is run.

The upstream
[TYK2 alchemical sample](https://github.com/NVIDIA/cuPotential/tree/v0.3.5/plugins/openmm/samples/alchemical)
lives in `plugins/openmm/samples/alchemical/` in a separate cuPotential source
checkout. Use a checkout matching the selected library release. The sample
compares a prepared window, changes charge/soft-core parameters and performs
short NVT dynamics. It is useful as an API example, but does not supply a complete
RBFE experiment. A subsequent free-energy calculation needs a defined
thermodynamic cycle, appropriate sampling, overlap/convergence analysis, an
estimator and relevant corrections. Preserve or disclose any change in tail
averaging when lowering the user's original forces.

Sources:

- [cuPotential custom interaction guide](https://docs.nvidia.com/cuda/cupotential/nonbonded-force).
- Gapsys et al. (2012), [soft-core potential construction](https://www.mpinat.mpg.de/634467/Gapsys_2012_JCTC_8_2373-2382.pdf): intermediate states can introduce numerical problems even when endpoints appear reasonable.
- Mey et al. (2020), [alchemical best practices](https://livecomsjournal.org/index.php/livecoms/article/view/v2i1e18378): sampling, analysis and protocol requirements beyond an expression update.
