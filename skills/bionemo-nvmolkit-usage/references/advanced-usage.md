# Advanced nvMolKit usage

## Configuration

Configuration objects expose operation-specific GPU and CPU execution controls.

### `HardwareOptions` (ETKDG, MMFF, UFF)

`from nvmolkit.types import HardwareOptions`. Passed via `hardwareOptions=` to `EmbedMolecules`, `MMFFOptimizeMoleculesConfs`, `UFFOptimizeMoleculesConfs`, and the `BatchedForcefield` constructors. Every field has an "auto" sentinel; the defaults are usually fine.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `preprocessingThreads` | int | `-1` (all visible CPUs) | CPU threads for preprocessing |
| `batchSize` | int | `-1` (auto-tuned) | Number of conformers per GPU batch |
| `batchesPerGpu` | int | `-1` (auto) | Concurrent batches per GPU; must be `>0` or `-1` |
| `gpuIds` | `list[int]` | `[]` (all visible GPUs) | Specific device ordinals to target |

Passing a `gpuIds` entry for a device that isn't visible raises `RuntimeError: invalid device ordinal`. For finding good values automatically across a representative sample, see `nvmolkit.autotune` (requires the `optuna` extra); each `tune_*` function returns a `TuneResult` whose `best_config` is a fully-populated `HardwareOptions` ready to pass back into the real call.

`HardwareOptions` round-trips through `to_dict()` / `from_dict()` for persisting tuned configs to disk.

### `SubstructSearchConfig` (substructure search)

`from nvmolkit.substructure import SubstructSearchConfig`. Passed via `config=` to `hasSubstructMatch`, `countSubstructMatches`, and `getSubstructMatches`.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `batchSize` | int | `1024` | (target, query) pairs per GPU batch |
| `workerThreads` | int | `-1` (auto) | GPU runner threads per GPU |
| `preprocessingThreads` | int | `-1` (auto) | CPU threads for preprocessing |
| `maxMatches` | int | `0` (unlimited) | Max matches returned per (target, query) pair |
| `uniquify` | bool | `False` | Drop duplicate matches that differ only in atom enumeration order |
| `gpuIds` | `list[int] \| None` | `None` (current device only) | Specific device ordinals to target |

Substructure search currently does not support chirality-aware matching, enhanced stereochemistry, or other advanced RDKit `SubstructMatchParameters` options.

### `MCSConfig` (maximum common substructure search)

`from nvmolkit.mcs import MCSConfig`. Pass it via `config=` to `findMCS`.
The fields `batchSize`, `workerThreads`, `preprocessingThreads`,
`executorsPerRunner`, and `gpuIds` control chunking and dispatch. Defaults
autoselect execution settings; an empty `gpuIds` list uses the current device.
`MCSConfig` supports `to_dict()` / `from_dict()` and can also be persisted with
`nvmolkit.autotune.save()` / `load()`.

## Custom forcefield options and constraints

Reach for `MMFFBatchedForcefield` / `UFFBatchedForcefield` instead of the one-shot `MMFFOptimizeMoleculesConfs` / `UFFOptimizeMoleculesConfs` when you need any of:

- Custom `maxIters` / `forceTol` per call
- Per-molecule `nonBondedThreshold` (MMFF) or `vdwThreshold` (UFF), or per-molecule `ignoreInterfragInteractions`
- Per-molecule `MMFFMolProperties` objects (e.g. MMFF94s vs MMFF94)
- Distance, position, angle, or torsion constraints
- Standalone `compute_energy()` / `compute_gradients()` without minimization

```python
from rdkit.Chem import AddHs, MolFromSmiles
from rdkit.Chem.rdDistGeom import EmbedMultipleConfs
from nvmolkit.batchedForcefield import MMFFBatchedForcefield

mols = [AddHs(MolFromSmiles(smi)) for smi in ["CCO", "CCCCCC"]]
for mol in mols:
    EmbedMultipleConfs(mol, numConfs=5)

ff = MMFFBatchedForcefield(
    mols,
    nonBondedThreshold=[100.0, 20.0],
    ignoreInterfragInteractions=True,
)

ff[0].add_position_constraint(0, max_displ=0.1, force_constant=50.0)
ff[1].add_distance_constraint(0, 4, relative=False, min_len=1.8, max_len=2.2, force_constant=25.0)

energies, converged = ff.minimize(maxIters=500, forceTol=1e-4)
for mol, mol_energies, mol_converged in zip(mols, energies, converged):
    print(mol.GetNumConformers(), mol_energies, mol_converged)
```

All conformers of each input molecule are minimized in one batch. Constraints attached via `ff[i].add_*_constraint(...)` apply to every conformer of molecule `i`; constraint setters mark the wrapper dirty and the native forcefield rebuilds on the next call. Pass `output=CoordinateOutput.DEVICE` to `.minimize(...)` to keep optimized coordinates on the GPU (`Device3DResult`) instead of writing them back into RDKit conformers. UFF is the same shape: `UFFBatchedForcefield(mols, vdwThreshold=..., ...)`.
