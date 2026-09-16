---
name: nvmolkit-usage
description: >-
  Use when writing or debugging nvMolKit Python code for GPU-accelerated RDKit
  fingerprints, similarity, conformers, clustering, and molecular searches.
license: Apache-2.0
metadata:
  author: Kevin Boyd (@scal444)
  owner: Kevin Boyd (@scal444)
  risk-tier: skill
  tags: [cheminformatics, rdkit, cuda]
---

# nvMolKit usage

## Purpose

GPU-accelerated, batched implementations of common RDKit operations. APIs mirror RDKit where possible but are batch-oriented: they take lists of `rdkit.Chem.Mol` (or lists of fingerprints) and process them in parallel on one or more GPUs. nvMolKit links against RDKit at build time; inputs and outputs are real RDKit `Mol` objects.

This skill covers the installed Python API. Building nvMolKit from source is out of scope.

## Where nvMolKit does well

Reach for nvMolKit when:

- The workload is **a large batch of molecules** processed together (typically thousands or more).
- The metric is **throughput / total wall time across the batch**, not per-molecule latency.
- The same operation is **repeated identically** across the batch (fingerprinting a library, embedding/minimizing many conformers, bulk pairwise similarity), so the GPU stays saturated.

## Requirements

- An NVIDIA GPU with compute capability 7.0 (V100) or higher
- A CUDA driver compatible with CUDA 12.6+.
- A working `torch` install with CUDA support (nvMolKit returns GPU tensors via `torch`'s CUDA array interface).

When helping with installation, make the user choose a PyTorch CUDA backend that the host driver supports before installing nvMolKit. nvMolKit's PyPI wheels are built with CUDA Toolkit 12.9 and depend on CUDA 12 runtime packages, but pip/uv can still select a CUDA 13 PyTorch wheel unless the install command says otherwise.

- Conda: prefer conda-forge `pytorch-gpu`; pin `cuda-version=12.6` or another CUDA version supported by the driver.
- pip: send the user to the [PyTorch install selector](https://pytorch.org/get-started/locally/) or [previous-versions page](https://pytorch.org/get-started/previous-versions/) to install `torch` for a CUDA 12.x backend before installing nvMolKit.
- uv: install nvMolKit with an explicit backend, e.g. `uv pip install --torch-backend=cu128 nvmolkit`.

## Inputs

- Required: choose an operation and supply molecules or fingerprints from the user's code or molecular dataset. Parse SMILES with RDKit and reject failed parses (`None`).
- Molecular operations use RDKit `Mol` objects. Add hydrogens for ETKDG; minimization and conformer comparisons need existing conformers.
- Fingerprint similarity takes packed `AsyncGpuResult`, torch tensors, or NumPy arrays: one molecule per row, with `int32` or `uint32` words.
- Optional: take conformer counts, fingerprint settings, cutoffs, output modes, and hardware options from the user's requested workflow; otherwise use the documented API defaults.

## Limitations

- CUDA is required; there is no CPU fallback. Use RDKit directly when CPU execution is needed.
- Plain RDKit is usually preferable for single-molecule work or operations that cannot be batched.
- ETKDG does not support custom bounds matrices, custom CPCI, coordinate maps, or separate-fragment embedding.
- Substructure search does not support chirality-aware matching, enhanced stereochemistry, or other advanced RDKit `SubstructMatchParameters` options.

## Instructions

1. Run the smoke test below before writing nvMolKit code.
2. Choose an API from the entry-point table and apply its input requirements.
3. Handle its result as described below; synchronize asynchronous GPU results before host reads.

### Verify the install before writing real code

```python
import nvmolkit
import torch
from rdkit import Chem
from nvmolkit.fingerprints import MorganFingerprintGenerator

print("nvmolkit:", nvmolkit.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

mols = [Chem.MolFromSmiles(smi) for smi in ["CCO", "c1ccccc1", "CC(=O)O"]]
fpgen = MorganFingerprintGenerator(radius=2, fpSize=1024)
result = fpgen.GetFingerprints(mols)
torch.cuda.synchronize()
fps = result.torch()
print("fps shape:", tuple(fps.shape), "dtype:", fps.dtype)
# Expected: shape (3, 32), dtype torch.int32  (1024 bits packed into 32 int32s per row)
```

If this fails, point the user at the [installation guide](https://nvidia-bionemo.github.io/nvMolKit/#installation) rather than guessing.

## Entry points

| Task | Module | Primary entry point |
|---|---|---|
| Morgan fingerprints | `nvmolkit.fingerprints` | `MorganFingerprintGenerator(radius, fpSize).GetFingerprints(mols)` |
| Bulk Tanimoto / cosine similarity | `nvmolkit.similarity` | `crossTanimotoSimilarity(...)`, `crossCosineSimilarity(...)`, plus `*MemoryConstrained` variants for results too large to fit in GPU memory |
| ETKDG conformer embedding | `nvmolkit.embedMolecules` | `EmbedMolecules(molecules, params, confsPerMolecule, ...)` |
| MMFF94 optimization (one-shot) | `nvmolkit.mmffOptimization` | `MMFFOptimizeMoleculesConfs(molecules, ..., minimizerKind=..., fireOptions=...)` |
| UFF optimization (one-shot) | `nvmolkit.uffOptimization` | `UFFOptimizeMoleculesConfs(molecules, ..., minimizerKind=..., fireOptions=...)` |
| Forcefield with custom options + constraints | `nvmolkit.batchedForcefield` | `MMFFBatchedForcefield(mols, properties=..., nonBondedThreshold=..., ignoreInterfragInteractions=..., hardwareOptions=...)`, `UFFBatchedForcefield(mols, vdwThreshold=..., ...)`. Per-molecule view `ff[i]` exposes `add_distance_constraint`, `add_position_constraint`, `add_angle_constraint`, `add_torsion_constraint`. Methods: `.compute_energy()`, `.compute_gradients()`, `.minimize(maxIters, forceTol, minimizerKind=..., fireOptions=...)` |
| Pairwise conformer RMSD | `nvmolkit.conformerRmsd` | `GetConformerRMSMatrix(mol)`, `GetConformerRMSMatrixBatch(mols)` |
| Torsion Fingerprint Deviation (TFD) | `nvmolkit.tfd` | `GetTFDMatrix(mol)`, `GetTFDMatrices(mols)` |
| Butina clustering | `nvmolkit.clustering` | `butina(distance_matrix, cutoff)` (precomputed matrix), `fused_butina(fingerprints, cutoff)` (memory-efficient, on-the-fly); both support explicit RDKit and device output modes |
| Substructure search | `nvmolkit.substructure` | `hasSubstructMatch`, `countSubstructMatches`, `getSubstructMatches` |
| Maximum common substructure | `nvmolkit.mcs` | `findMCS(mols, ...)` for all pairs, explicit pairs, or two paired molecule lists |
| Hardware tuning (batch size, GPU IDs) | `nvmolkit.types` | `HardwareOptions(...)` passed to ETKDG / MMFF / UFF |
| Optional autotuning | `nvmolkit.autotune` | `tune_embed_molecules`, `tune_mmff_optimize`, `tune_uff_optimize`, `tune_batched_forcefield`, `tune_substructure`, `tune_mcs`. Requires the `optuna` package |

## Result types and execution model

Two return shapes carry GPU-resident output, depending on what the operation produces.

### `AsyncGpuResult`

Used by operations that return a single flat tensor (fingerprints, similarity matrices, RMSD/TFD vectors, Butina inputs). Key behaviors:

- Asynchronous. The kernel may not have completed when the call returns.
- `result.torch()` returns a zero-copy `torch.Tensor` on the GPU. Caller is responsible for synchronizing before reading values on the host.
- `result.numpy()` synchronizes and returns a CPU numpy array.
- Exposes `__cuda_array_interface__`, so it can be passed directly into other nvMolKit functions (e.g. fingerprints → similarity) with no host round-trip.

#### CUDA stream control

A subset of the `AsyncGpuResult`-returning APIs accept an optional `stream: torch.cuda.Stream | None = None` argument so callers can submit nvMolKit work to a non-default stream and overlap it with their own kernels. When omitted, the call uses the current torch stream.

APIs that take a `stream` argument:

- `MorganFingerprintGenerator.GetFingerprints`
- `crossTanimotoSimilarity`, `crossCosineSimilarity`, and their `*MemoryConstrained` variants
- `butina`, `fused_butina`
- `GetConformerRMSMatrix`, `GetConformerRMSMatrixBatch`

Other APIs (ETKDG, MMFF/UFF optimization, TFD, substructure search, MCS) are synchronous to the caller — no stream plumbing needed.

Typical pattern:

```python
import torch
from rdkit import Chem
from nvmolkit.fingerprints import MorganFingerprintGenerator
from nvmolkit.similarity import crossTanimotoSimilarity

stream = torch.cuda.Stream()
fpgen = MorganFingerprintGenerator(radius=2, fpSize=1024)
mols = [Chem.MolFromSmiles(smi) for smi in ["CCO", "c1ccccc1", "CC(=O)O"]]

with torch.cuda.stream(stream):
    fps = fpgen.GetFingerprints(mols, stream=stream)
    sim = crossTanimotoSimilarity(fps, stream=stream)
stream.synchronize()
print(sim.torch())
```

### `Device3DResult`

Used by ETKDG embedding and MMFF/UFF optimization (one-shot and `BatchedForcefield`) when called with `output=CoordinateOutput.DEVICE`. The GPU-resident equivalent of writing conformers back to `Mol` objects. Fields:

- `values`: `AsyncGpuResult` of shape `(total_atoms, 3)` float64. Concatenated conformer coordinates in CSR-style layout.
- `atom_starts`, `mol_indices`, `conf_indices`: `AsyncGpuResult` int32 buffers describing the layout (`values[atom_starts[i]:atom_starts[i+1]]` is conformer `i`'s atoms).
- `energies`, `converged`: `AsyncGpuResult` buffers populated only for MMFF/UFF minimization (not for plain ETKDG).
- `gpu_id`: device the buffers live on. The `targetGpu` argument on each API picks this; `targetGpu=-1` uses the default consolidation device.
- `.per_molecule()` returns nested `list[list[torch.Tensor]]` of per-conformer views; `.dense(pad_value=nan)` materializes a padded `(n_mols, max_confs, max_atoms, 3)` tensor.

The default mode (`CoordinateOutput.RDKIT_CONFORMERS`) still writes optimized coordinates back into each `Mol` and returns Python lists of energies/convergence flags. Reach for `CoordinateOutput.DEVICE` when chaining downstream GPU work (e.g. ETKDG → MMFF → similarity scoring) without host round-trips.

### `MCSBatchResult`

`findMCS` is synchronous and returns an `MCSBatchResult` backed by CPU NumPy
arrays. Results are always flat: `result[k]` (or `result.get_result(k)`)
materializes the result at pair position `k`, not generally the result for
molecule `k`. Use `result.pairs[k]` to identify that pair. In `all_pairs` mode
these are the generated pairs over `mols`; in `pairs` mode they exactly preserve
the supplied pair sequence; in `paired_lists` mode item `k` compares `mols[k]`
with `mols_b[k]`, while `result.pairs[k]` uses the combined-table indices
`(k, len(mols) + k)`. Each `MCSResult` has `pair`, `num_atoms`, `num_bonds`,
`canceled`, `atom_mapping`, and `bond_mapping`; the two columns of each mapping
index the first and second molecule of that result pair, respectively.

## Configuration

For ETKDG, forcefield, substructure, or MCS tuning, read the
[advanced configuration reference](references/advanced-usage.md#configuration).
It lists configuration fields, defaults, GPU selection, and autotuning APIs.

## Examples

### Morgan fingerprints + bulk Tanimoto similarity

```python
import torch
from rdkit import Chem
from nvmolkit.fingerprints import MorganFingerprintGenerator
from nvmolkit.similarity import crossTanimotoSimilarity

smiles = ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCOCC"]
mols = [Chem.MolFromSmiles(smi) for smi in smiles]

fpgen = MorganFingerprintGenerator(radius=2, fpSize=1024)
fps = fpgen.GetFingerprints(mols)

sim = crossTanimotoSimilarity(fps)
torch.cuda.synchronize()
print(sim.torch())
```

Inputs are `list[Mol]`. Output of `GetFingerprints` is an `AsyncGpuResult` wrapping an `(n_mols, fpSize / 32)` int32 tensor of packed bits. Pass it straight into `crossTanimotoSimilarity` for an `(n, n)` similarity matrix; pass two fingerprint sets for an `(n, m)` cross-matrix. For sets too large to materialize on the GPU, use `crossTanimotoSimilarityMemoryConstrained` (chunked compute, returns numpy on CPU).

### ETKDG conformer embedding

```python
from rdkit.Chem import AddHs, MolFromSmiles
from rdkit.Chem.rdDistGeom import ETKDGv3
from nvmolkit.embedMolecules import EmbedMolecules

mols = [AddHs(MolFromSmiles(smi)) for smi in ["C1CCCCC1", "C1CCCCC2CCCCC12", "COO"]]
params = ETKDGv3()

EmbedMolecules(mols, params, confsPerMolecule=10, maxIterations=-1)

for mol in mols:
    print(mol.GetNumConformers())
```

Inputs are sanitized `list[Mol]` with hydrogens added (`AddHs`). Conformers are added in-place; see Limitations for unsupported embedding options.

### MMFF94 minimization of a batch of conformers

```python
from rdkit.Chem import AddHs, MolFromSmiles
from rdkit.Chem.rdDistGeom import ETKDGv3
from nvmolkit.embedMolecules import EmbedMolecules
from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs

mols = [AddHs(MolFromSmiles(smi)) for smi in ["CCO", "CCN", "c1ccccc1"]]
params = ETKDGv3()
EmbedMolecules(mols, params, confsPerMolecule=5)

energies = MMFFOptimizeMoleculesConfs(mols, maxIters=500)
for mol, mol_energies in zip(mols, energies):
    print(mol.GetNumConformers(), mol_energies)
```

Inputs are `list[Mol]` with conformers already populated (typically by ETKDG, RDKit's `EmbedMultipleConfs`, or a prior nvMolKit call). Coordinates are updated in place; the return is `list[list[float]]` of optimized energies aligned with the input molecule order and conformer index. UFF is identical in shape: swap in `from nvmolkit.uffOptimization import UFFOptimizeMoleculesConfs`.

BFGS is the default minimizer. To use FIRE, pass `minimizerKind="FIRE"`;
optionally customize it with a
`nvmolkit.types.FireOptions` instance through `fireOptions=`. The one-shot MMFF
and UFF functions and both batched-forcefield `.minimize()` methods accept the
same selector.

If any input molecule is `None` or lacks MMFF/UFF atom types, the call raises `ValueError`. The exception's `args[1]` is a dict with keys `"none"` and `"no_params"` listing the offending indices - useful for filtering a noisy input set.

### Conformer RMSD and Butina clustering

```python
import torch
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMultipleConfs
from nvmolkit.clustering import ButinaOutputMode, butina
from nvmolkit.conformerRmsd import GetConformerRMSMatrixBatch

mols = [Chem.AddHs(Chem.MolFromSmiles(smi)) for smi in ["CCCCCC", "c1ccccc1"]]
for mol in mols:
    EmbedMultipleConfs(mol, numConfs=10)

# Remove hydrogens after embedding for heavy-atom RMSD.
heavy_mols = [Chem.RemoveHs(mol) for mol in mols]

# Default RMSD output is RDKit-compatible condensed lower-triangle form.
condensed = GetConformerRMSMatrixBatch(heavy_mols)

# Butina expects a square distance matrix, so request square GPU tensors.
square = GetConformerRMSMatrixBatch(heavy_mols, output_format="square")
results = [
    butina(distance_matrix, cutoff=0.5, output=ButinaOutputMode.DEVICE)
    for distance_matrix in square
]

torch.cuda.synchronize()
for result in results:
    print(result.cluster_ids.torch().cpu().tolist())
```

Both Butina functions return GPU-resident results by default:

- The default, `output=ButinaOutputMode.DEVICE`, returns cluster IDs, centroids, and sizes.
- `output=ButinaOutputMode.RDKIT` returns RDKit cluster tuples on the host. The first element of each cluster is its centroid.

The device output fields are `AsyncGpuResult` objects. Use `.torch()` to access
their CUDA tensors without a host copy or `.numpy()` to synchronize and copy a
field to the host.

`GetConformerRMSMatrix(mol)` and `GetConformerRMSMatrixBatch(mols)` default to `output_format="condensed"`, returning `AsyncGpuResult` objects that wrap RDKit-style flat vectors of length `N * (N - 1) // 2`. Use `output_format="square"` when chaining into `butina()` or any other API that expects an `N x N` distance matrix. Both forms live on the GPU; call `.numpy()` on condensed results or synchronize before moving square tensors to the CPU.

### Atom-Atom Path similarity and directed sphere exclusion clustering

Atom-Atom Path (AAP) similarity with directed sphere exclusion (DISE)
clustering provides device and RDKit-style output modes:

```python
from rdkit import Chem
from nvmolkit.clustering import DISEOutputMode, aap_dise

molecules = [Chem.MolFromSmiles(smiles) for smiles in ["CCCC", "CCCO", "CCOC"]]
device_result = aap_dise(molecules)
rdkit_clusters = aap_dise(molecules, output=DISEOutputMode.RDKIT)
```

`device_result` has `cluster_ids`, `centroids`, and `cluster_sizes` fields;
cluster IDs are zero-based and contiguous. `DISEOutputMode.RDKIT` describes
the centroid-first RDKit cluster representation, not an RDKit implementation
of the AAP+DISE algorithm. The current DISE control loop synchronizes before
returning either mode; `DEVICE` describes the stable schema and where the
result resides, not asynchronous execution of the overall call.

### Maximum common substructure search

```python
from rdkit import Chem
from nvmolkit.mcs import findMCS

mols = [Chem.MolFromSmiles(smi) for smi in ["CCO", "CCN", "c1ccccc1O"]]
result = findMCS(mols, mode="pairs", pairs=[(0, 1), (0, 2)])

for pair_idx, pair in enumerate(result.pairs):
    item = result[pair_idx]
    print(pair, item.num_atoms, item.num_bonds, item.atom_mapping)
```

The default `mode="all_pairs"` generates the upper triangle including the
diagonal. `mode="pairs"` preserves an explicit pair list exactly, including
duplicates and reversed pairs. `mode="paired_lists"` zips `mols` with an
equally sized `mols_b`. Timeouts are per pair; inspect `item.canceled` because a
timed-out result can contain the best partial MCS found.

Matching options include `atom_compare`, `bond_compare`, valence/formal-charge
matching, and atom/bond ring-only matching. Unsupported RDKit fMCS options
raise instead of silently changing semantics. For repeated representative
explicit-pair workloads, `nvmolkit.autotune.tune_mcs` returns a `TuneResult`.
Its `best_config` is the tuned `MCSConfig` to pass to
`findMCS(..., config=result.best_config)`.

### Custom forcefield options + constraints (`BatchedForcefield`)

For per-molecule forcefield settings, geometric constraints, or separate
energy and gradient calls, read the
[advanced forcefield recipe](references/advanced-usage.md#custom-forcefield-options-and-constraints).

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `torch.cuda.is_available()` is false | GPU access, driver compatibility, or the torch CUDA build is missing | Check the GPU and driver, then follow the installation guidance above to select a compatible torch build. |
| `RuntimeError: invalid device ordinal` | A requested `gpuIds` entry is not visible | Use device indices below `torch.cuda.device_count()` or the API's documented GPU defaults. |
| An RDKit option is rejected | The option is unsupported by that nvMolKit API | Use supported options only if they preserve the requested behavior; otherwise use RDKit for that operation. |

## Going deeper

- Full feature list, API reference, and guides: <https://nvidia-bionemo.github.io/nvMolKit/>
- What changed in each release: <https://nvidia-bionemo.github.io/nvMolKit/changelog.html>
- Worked examples (Jupyter notebooks): the [examples/ directory](https://github.com/NVIDIA-BioNeMo/nvMolKit/tree/main/examples) in the GitHub repo
