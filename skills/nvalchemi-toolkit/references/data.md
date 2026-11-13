# Data: atomic systems, batches, and storage

This note covers `AtomicData` and `Batch`, conversion from ASE, Zarr storage,
and streaming data into a model or a training loop.

## Where to read more

| Topic | Docs page | Example | Installed source |
| --- | --- | --- | --- |
| `AtomicData`, `Batch`, field levels, groups | [data.md][doc-data] | [01_data_structures.py][ex-data] | `nvalchemi/data/atomic_data.py`, `nvalchemi/data/batch.py` |
| Writer, reader, dataset, loader | [datapipes.md][doc-pipes] | [02_trajectory_zarr_io.py][ex-zarr] | `nvalchemi/data/datapipes/` |
| Chunking, compression, throughput | [zarr_compression.md][doc-zarr] | none | `nvalchemi/data/datapipes/backends/zarr.py`, `nvalchemi/data/io_test.py` |
| Trajectory sinks during dynamics | [dynamics_sinks.md][doc-sinks] | [02_trajectory_zarr_io.py][ex-zarr] | `nvalchemi/dynamics/sinks.py` |
| ASE conversion | [data.md][doc-data] (ASE section) | [03_ase_integration.py][ex-ase] | `AtomicData.from_atoms` in `atomic_data.py` |
| Units and sign conventions | [conventions.md][doc-conv] | none | `nvalchemi/dynamics/_units.py` |

API reference sources: [docs/modules/data.rst][api-data].

## Which component for which need

| Need | Component |
| --- | --- |
| One structure | `AtomicData(positions=..., atomic_numbers=..., ...)` |
| Many structures in one GPU pass | `Batch.from_data_list([...])` |
| Structures from ASE or extxyz | `AtomicData.from_atoms(atoms)` per frame, then `Batch` |
| Tag images of one path, or replicas of one system | `batch.set_group_layout(group_idx)` |
| Per-group reductions and broadcasts | `batch.group_layout` (a `GroupLayout`: `reduce_all`, `reduce_any`, `broadcast`, `node_to_group`) |
| Tile a periodic structure | `make_supercell(data, repeats)` (`nvalchemi.data.transforms`) |
| Keep a float64 label in a float32 dataset | `AtomicData.precision_preserving_keys = frozenset({"energy"})`, see [data.md][doc-data] |
| Batch from trusted raw tensors, no validation | `Batch.from_raw_dicts([...], device=...)` |
| Fixed-capacity buffer for moving graphs | `Batch.empty(num_systems=..., num_nodes=..., num_edges=..., template=...)`, then `buffer.put(src, mask)` |
| Save structures or labels to disk | `AtomicDataZarrWriter(path).write(...)` |
| Add more frames to an existing store | `AtomicDataZarrWriter(path).append(...)` |
| Raw tensors per sample | `AtomicDataZarrReader(path)` |
| Check a store's fields or repair state | `reader.schema()`, `reader.level_sizes()`, `reader.check_integrity()` |
| `AtomicData` per sample, device transfer, prefetch | `Dataset(reader, device=...)` |
| Batches for training or inference | `DataLoader(dataset, batch_size=...)` |
| Small store that fits in host memory | `InMemoryDataset(reader=reader, device=device)` |
| A few hundred frames already in memory | `InMemoryDataset(batch, device=device)` |
| Several stores behind one index | `MultiDataset(ds_a, ds_b)` with `MultiDatasetBatchSampler` |
| Snapshots during dynamics | `ZarrData` or `HostMemory` sink with `SnapshotHook` |
| A storage format the toolkit lacks | Subclass `Reader` (`nvalchemi/data/datapipes/backends/base.py`) |

Imports: `from nvalchemi.data import AtomicData, Batch` and
`from nvalchemi.data.datapipes import AtomicDataZarrWriter,
AtomicDataZarrReader, Dataset, DataLoader`.

## How the pieces connect

```text
ASE Atoms / tensors -> AtomicData -> Batch.from_data_list -> model or dynamics
                                  \-> AtomicDataZarrWriter -> store.zarr
store.zarr -> AtomicDataZarrReader -> Dataset -> DataLoader -> Batch
```

- `AtomicData` holds one graph. Fields have a level: node (`positions`,
  `atomic_numbers`, `atomic_masses`, `forces`, `velocities`), edge
  (`neighbor_list`, `shifts`), and system (`cell`, `pbc`, `energy`,
  `stress`). System fields carry a leading dimension of 1, for example
  `energy` is `[1, 1]` and `cell` is `[1, 3, 3]`.
- `Batch` concatenates graphs. `batch.batch_idx` maps each atom to its graph;
  `batch.num_nodes_per_graph` gives atom counts. Use them for per-graph
  reductions instead of Python loops.
- The `DataLoader` yields `Batch` objects. `Dataset.load_batches(...)` is the
  explicit batch read.
- Dynamics sinks (`ZarrData`) write stores that the same reader reads back.

## Keep hot paths on `Batch`

`AtomicData` is a Pydantic model. It validates every field each time you
build one. Use it to get checked input at setup: from ASE, from user
tensors, or for a few test structures. Do not build it inside a step loop, a
data-loading loop, or per-sample code in a training or dynamics loop.

Hot paths stay on `Batch`:

- Work on batch tensors. Reduce per graph with `batch.batch_idx` and
  `batch.num_nodes_per_graph`.
- Move graphs between batches with `Batch.empty(...)` buffers,
  `buffer.put(src, mask)`, and `batch.index_select(idx)`. `put` appends after
  the graphs already stored. Read `copied_mask` to see which source graphs
  fit. Since 0.3.0 it needs no `dest_mask`. That argument is ignored and
  warns.
- Build batches from trusted tensors with `Batch.from_raw_dicts(...)`. It
  skips `AtomicData` construction and validation.
- Read toolkit-written stores with `Dataset(..., skip_validation=True)`.
- Do not call `batch.get_data(i)`, `batch.to_data_list()`, or
  `batch.append_data(...)` in a loop. Each one builds `AtomicData` objects.
  Use them at setup or in the final report.

## Field names that trip agents up

**Masses are `atomic_masses`.** There is no `masses` field or alias.
`batch.masses` raises `AttributeError`. `AtomicData` fills `atomic_masses`
from the periodic table when you do not pass it. Integrator and thermostat
code must read `batch.atomic_masses` (shape `[V]`, unsqueeze to `[V, 1]` to
divide forces).

**Labels stored under other names.** `from_atoms` reads `energy` from
`atoms.info` and `forces` from `atoms.arrays` by default. When the file uses
other names, pass them:
`AtomicData.from_atoms(atoms, energy_key="dft_energy", forces_key="dft_forces")`.
Do not set `data.energy` or `data.forces` by hand. For a few hundred labelled
frames, build one `Batch` and wrap it in
`InMemoryDataset(batch, device=device)`. You do not need a Zarr store.

**Velocities are not read from files.** `AtomicData.from_atoms` copies
positions, numbers, cell, pbc, and optional energy, forces, stress, and
charges. It does not read velocities or momenta, even when the ASE object or
the extxyz file stores them. The result has zero velocities. Set them
yourself after conversion:

```python
import torch
from ase.io import read
from nvalchemi.data import AtomicData

frames = read("input.extxyz", index=":")
data_list = []
for atoms in frames:
    data = AtomicData.from_atoms(atoms)
    # ASE velocities are in Å per ASE time unit, which equals the toolkit's
    # sqrt(eV/amu); get_velocities() derives them from stored momenta / masses.
    v = torch.as_tensor(atoms.get_velocities(), dtype=data.positions.dtype)
    data.velocities = v
    data_list.append(data)
```

If the file stores velocities in Å/fs instead, multiply by the toolkit time
factor (`FS_PER_INTERNAL_TIME` in `nvalchemi/dynamics/_units.py`, about
10.18). Check the file's convention before you convert. To draw thermal
velocities instead, use `nvalchemi.dynamics.initialize_velocities`.

**Dynamics output buffers.** Integrators write `forces` and `energy` in place.
`from_atoms` leaves them `None` when the file has no labels. Most engines
allocate them on the first step, but scripts that read `batch.forces` before
the first step, and some strategies (enhanced sampling needs `stress` for
periodic biased runs), expect them. The safe pattern from the ASE example:

```python
n = data.num_nodes
data.forces = torch.zeros(n, 3)
data.energy = torch.zeros(1, 1)
```

**Atom categories.** `from_atoms` does not set `atom_categories`. Hooks that
act on a subset (for example `FreezeAtomsHook`) read
`atom_categories == AtomCategory.SPECIAL.value` (from `nvalchemi._typing`).
Set it from ASE tags or indices after conversion.

**Custom per-system or per-atom data.** A plain assignment of an unknown key
on a `Batch` (`batch.foo = t`) is stored as a system-level field, so a `[B, 1]`
tensor survives `to_data_list()` and sinks. For per-atom or per-edge data use
`data.add_node_property(name, t)` or `batch.add_key(name, values,
level="node")`, so the level is explicit.

**Groups.** `batch.set_group_layout(group_idx)` needs each group's graphs to
be contiguous. Set groups through it, not by assigning `batch.group_idx`.
The `batch.group_layout` property is cached: after an in-place edit such as
`batch.group_idx[3] = 2`, call `set_group_layout` again. `index_select()`
selects complete groups without repeated graphs; reordered groups must remain
contiguous. It normalizes labels and builds the selected batch's layout.
`append()` requires both batches grouped or both ungrouped and rebases labels.
Grouped batches reject `append_data()`. `put()` and `trim()`
accept whole-group masks and reject partial groups; `put()` copies only
complete groups that fit. See [data.md][doc-data] (Grouping graphs).

`append()` and `concatenate()` discard unused buffer rows by default.
`assume_compact=True` skips that step for speed and is only safe when inputs
have no padding; padded inputs then return padding rows as atoms. Use `put()`
to keep buffer capacity. `HostMemory.write()` trims padding before copying to CPU.

## Storage patterns

Write once, then append:

```python
from nvalchemi.data.datapipes import AtomicDataZarrWriter

writer = AtomicDataZarrWriter("frames.zarr")
writer.write(data_list[:100])     # creates the store; a second write() raises
writer.append(data_list[100:])    # adds to it
```

`write()` raises `FileExistsError` when the store exists. Pick `append()` or a
new path on purpose, so the script can be rerun.

Stream it back in batches:

```python
from nvalchemi.data.datapipes import AtomicDataZarrReader, DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = Dataset(AtomicDataZarrReader("frames.zarr"), device=device)
loader = DataLoader(dataset, batch_size=64, shuffle=True)
for batch in loader:
    ...
```

Use the same `device` for the model and the training strategy. For in-memory
structures, use `Batch.from_data_list(data_list).to(device)` once at setup.

To mix stores, wrap them in `MultiDataset` and pick the mixture with
`MultiDatasetBatchSampler.balanced(dataset, batch_size=..., epoch_policy=...)`:
`"min_size"` ends the epoch at the smallest store, `"max_size"` covers the
largest and resamples the others (`replacement=True`). For fixed ratios, pass
`samples_per_dataset` to `MultiDatasetBatchSampler`. Details in
[datapipes.md][doc-pipes].

Do not hand-write a loop that opens Zarr arrays and builds `AtomicData` per
sample. The reader, dataset, and loader handle offsets, device transfer, and
prefetching.

## Throughput tuning

Start here when a training or inference loop waits on data:

- `Dataset(..., skip_validation=True)` for stores written by the toolkit. It
  skips per-sample validation. Keep validation on for third-party stores.
- `DataLoader(..., pin_memory=True)` for CUDA targets.
- `prefetch_factor` fuses that many batches into one backend read. Shuffled
  reads gain most (16 to 64). Sequential reads need 2 to 4. `0` turns fused
  prefetch off.
- `num_workers=1` on `Dataset` is usually enough; concurrent Zarr reads
  contend.
- Chunk and shard sizes are set at write time with
  `ZarrWriteConfig(core=ZarrArrayConfig(...))`. `chunk_size` counts rows on
  the atom or edge axis, not structures.
- Measure with the `nvalchemi-io-test` command (`roundtrip` or `read`
  subcommands) and sweep `--prefetch-factor` before you change chunking.

The tuning guide has the numbers: [zarr_compression.md][doc-zarr].

## Common mistakes

- Looping over structures and running one model call or one dynamics run per
  structure. Build one `Batch`.
- Reading `masses` instead of `atomic_masses`.
- Assuming `from_atoms` keeps velocities from an extxyz file.
- Writing to an existing store with `write()`.
- Building `energy` as shape `[1]` or `[B]`. The batch convention is
  `[B, 1]`; single-graph `AtomicData` uses `[1, 1]`.
- Mixing devices: build on CPU, then move the whole batch with `.to(device)`.
- Building `AtomicData` per sample or per step in a hot loop. Stay on
  `Batch` (see "Keep hot paths on `Batch`").
- Passing `dest_mask=` to `Batch.put`. It is deprecated and ignored.
- Source and buffer with different dtypes for one field. `Batch.put` needs
  the same dtype on both sides.

## How to check the result

- Print `batch.num_graphs`, `batch.num_nodes`, and
  `batch.num_nodes_per_graph`; they must match the input frames.
- Round-trip: `Batch.from_data_list(batch.to_data_list())` keeps positions,
  numbers, and custom keys.
- After writing a store, open it with `AtomicDataZarrReader` and compare
  `len(reader)` and one sample's positions with the source.
- After converting from a file, compare `data.velocities` with the file's
  velocities for one frame, and kinetic energy
  `0.5 * (m * v**2).sum()` with the file's value if it stores one.

[doc-data]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/data.md
[doc-pipes]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/datapipes.md
[doc-zarr]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/zarr_compression.md
[doc-sinks]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/dynamics_sinks.md
[doc-conv]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/userguide/about/conventions.md
[api-data]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/docs/modules/data.rst
[ex-data]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/basic/01_data_structures.py
[ex-zarr]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/intermediate/02_trajectory_zarr_io.py
[ex-ase]: https://github.com/NVIDIA/nvalchemi-toolkit/blob/v0.3.0/examples/basic/03_ase_integration.py
