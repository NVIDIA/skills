#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a stock OpenMM CPU control; never evaluates cuPotential or CUDA.

Accepts the bundled synthetic system.json model. Uses only the selected
interpreter's OpenMM installation and the standard library. Installs nothing.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def context_from(system, integrator, platform):
    import openmm as mm
    selected = mm.Platform.getPlatformByName(platform)
    properties = {"Threads": "1"} if platform == "CPU" else {}
    return mm.Context(system, integrator, selected, properties)


def build(model, ensemble):
    import openmm as mm
    system = mm.System()
    for mass in model["masses_dalton"]:
        system.addParticle(mass)
    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*v) for v in model["box_vectors_nm"]])
    if model.get("virtual_sites"):
        raise ValueError("This synthetic-model helper does not implement virtual sites")
    for row in model["constraints"]:
        system.addConstraint(*row["particles"], row["distance_nm"])
    spec = model["nonbonded"]
    force = mm.NonbondedForce()
    if spec["method"] not in {"PME", "LJPME"}:
        raise ValueError("Expected the supplied PME/LJPME model")
    force.setNonbondedMethod(getattr(mm.NonbondedForce, spec["method"]))
    force.setCutoffDistance(spec["cutoff_nm"])
    force.setEwaldErrorTolerance(spec["ewald_error_tolerance"])
    force.setUseSwitchingFunction(spec["switching"])
    if spec["switching"]:
        force.setSwitchingDistance(spec["switching_distance_nm"])
    force.setUseDispersionCorrection(spec["dispersion_correction"])
    force.setExceptionsUsePeriodicBoundaryConditions(spec["exceptions_use_periodic_boundary_conditions"])
    for row in spec["particles"]:
        force.addParticle(row["charge_e"], row["sigma_nm"], row["epsilon_kj_mol"])
    for row in spec["exceptions"]:
        force.addException(*row["particles"], row["charge_product_e2"],
                           row["sigma_nm"], row["epsilon_kj_mol"])
    system.addForce(force)
    if model["com_frequency"]:
        system.addForce(mm.CMMotionRemover(model["com_frequency"]))
    if ensemble == "npt":
        row = model["npt"]
        barostat = mm.MonteCarloBarostat(row["pressure_bar"], row["temperature_kelvin"], row["frequency"])
        barostat.setRandomNumberSeed(row["seed"])
        system.addForce(barostat)
    integrator = mm.LangevinMiddleIntegrator(
        model["temperature_kelvin"], model["friction_per_ps"], model["timestep_ps"])
    integrator.setRandomNumberSeed(model["integrator_seed"])
    return system, integrator


def snapshot(context):
    from openmm import unit
    state = context.getState(getPositions=True, getVelocities=True, getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = [list(v) for v in state.getForces().value_in_unit(unit.kilojoule_per_mole / unit.nanometer)]
    if not math.isfinite(energy) or not all(math.isfinite(v) for row in forces for v in row):
        raise ValueError("Nonfinite stock CPU energy/forces")
    return {
        "positions_nm": [list(v) for v in state.getPositions().value_in_unit(unit.nanometer)],
        "velocities_nm_ps": [list(v) for v in state.getVelocities().value_in_unit(unit.nanometer / unit.picosecond)],
        "box_vectors_nm": [list(v) for v in state.getPeriodicBoxVectors().value_in_unit(unit.nanometer)],
        "time_ps": state.getTime().value_in_unit(unit.picosecond),
        "step_count": context.getStepCount(), "parameters": dict(context.getParameters()),
        "energy_kJ_mol": energy, "forces_kJ_mol_nm": forces,
    }


def restore(args):
    import openmm as mm
    directory = Path(args.output)
    if any((directory / name).exists() for name in (
        "restored.json", "continued.json", "restoration_comparison.json", "subprocess.json")):
        raise FileExistsError("Restore evidence already exists; start a new reference run")
    config = json.loads((directory / "run_config.json").read_text())
    system = mm.XmlSerializer.deserialize((directory / "system.xml").read_text())
    integrator = mm.XmlSerializer.deserialize((directory / "integrator.xml").read_text())
    context = context_from(system, integrator, config["platform"])
    context.loadCheckpoint((directory / "state.chk").read_bytes())
    restored = snapshot(context)
    write(directory / "restored.json", restored)
    before = json.loads((directory / "before_save.json").read_text())
    fields = json.loads((directory / "criteria.json").read_text())["exact_restoration_fields"]
    checks = {key: before[key] == restored[key] for key in fields}
    write(directory / "restoration_comparison.json", checks)
    if not all(checks.values()):
        raise ValueError("CPU physical-state restoration failed; see restoration_comparison.json")
    integrator.step(config["resume_steps"])
    continued = snapshot(context)
    write(directory / "continued.json", continued)
    if continued["step_count"] != before["step_count"] + config["resume_steps"]:
        raise ValueError("Unexpected resumed CPU step count")
    write(directory / "subprocess.json", {"pid": os.getpid(), "parent_pid": os.getppid(),
                                         "interpreter": sys.executable})


def run(args):
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        raw = Path(args.model).read_bytes()
        model = json.loads(raw)
        (directory / "input.json").write_bytes(raw)
        config = {"platform": args.platform, "ensemble": args.ensemble,
              "steps": args.steps, "resume_steps": args.resume_steps,
              "input_sha256": hashlib.sha256(raw).hexdigest(),
              "requested_cuda_precision": model["precision"],
              "cpu_precision_policy": "OpenMM CPU/Reference; CUDA Precision property is not applicable",
              "interpreter": sys.executable, "pid": os.getpid()}
        write(directory / "run_config.json", config)
        write(directory / "commands.json", {"initial": sys.argv, "restore": None})
        # Declare the physical-state comparison before observing any results.
        write(directory / "criteria.json", {"exact_restoration_fields": [
            "positions_nm", "velocities_nm_ps", "box_vectors_nm", "time_ps", "step_count", "parameters"],
            "future_bitwise_trajectory_claim": False})
        import openmm as mm
        config["openmm_version"] = mm.__version__
        write(directory / "run_config.json", config)
        system, integrator = build(model, args.ensemble)
        (directory / "system.xml").write_text(mm.XmlSerializer.serialize(system))
        (directory / "integrator.xml").write_text(mm.XmlSerializer.serialize(integrator))
        write(directory / "force_inventory.json", [type(f).__name__ for f in system.getForces()])
        context = context_from(system, integrator, args.platform)
        context.setPositions(model["positions_nm"])
        velocity = model["initial_velocities"]
        if velocity["method"] != "setVelocitiesToTemperature":
            raise ValueError("Unsupported initial velocity policy")
        context.setVelocitiesToTemperature(model["temperature_kelvin"], velocity["seed"])
        if velocity["apply_velocity_constraints"]:
            context.applyVelocityConstraints(integrator.getConstraintTolerance())
        write(directory / "initial.json", snapshot(context))
        integrator.step(args.steps)
        write(directory / "before_save.json", snapshot(context))
        (directory / "state.chk").write_bytes(context.createCheckpoint())
        # Release CPU PME resources before starting the independent process.
        del context, integrator, system
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "restore", "--output", str(directory.resolve())]
        write(directory / "commands.json", {"initial": sys.argv, "restore": command})
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        write(directory / "subprocess_result.json", {"returncode": result.returncode,
              "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise RuntimeError("Fresh-process CPU restore failed; see subprocess_result.json")
        write(directory / "status.json", {"status": "CPU_REFERENCE_PASS", "platform": args.platform,
              "steps": args.steps, "resumed_steps": args.resume_steps,
              "cupotential_cuda": "NOT_RUN", "cupotential_numerical_agreement": "NOT_TESTED",
              "future_bitwise_trajectory": "NOT_CERTIFIED"})
    except Exception as error:
        write(directory / "status.json", {"status": "CPU_REFERENCE_FAILED", "cupotential_cuda": "NOT_RUN",
                                          "error": f"{type(error).__name__}: {error}"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    execute = subparsers.add_parser("run")
    execute.add_argument("--model", required=True)
    execute.add_argument("--output", required=True, help="New directory; existing runs are preserved")
    execute.add_argument("--ensemble", choices=["nvt", "npt"], default="nvt")
    execute.add_argument("--platform", choices=["CPU", "Reference"], default="CPU")
    execute.add_argument("--steps", type=int, default=100)
    execute.add_argument("--resume-steps", type=int, default=40)
    resume = subparsers.add_parser("restore")
    resume.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.mode == "run" and (args.steps <= 0 or args.resume_steps < 0):
        parser.error("steps must be positive and resume-steps nonnegative")
    (run if args.mode == "run" else restore)(args)


if __name__ == "__main__":
    main()
