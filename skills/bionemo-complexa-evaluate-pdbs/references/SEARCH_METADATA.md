# Search Metadata Tags

Every sample produced by a search algorithm carries a **metadata tag** -- a string that encodes the algorithm, original sample identity, and full lineage of branching decisions.

> **Documentation Map**
> - Running a design? See [Inference Guide](INFERENCE.md).
> - Tuning YAML configs? See [Configuration Guide](CONFIGURATION_GUIDE.md).
> - Understanding metrics? See [Evaluation Guide](EVALUATION_METRICS.md).
> - Parameter sweeps? See [Sweep System](SWEEP.md).

The tag appears in:

- **Filenames**: each PDB is saved as `job_{id}_n_{len}_id_{idx}_{tag}.pdb`
- **CSV columns**: the `metadata_tag` column in results and top-samples CSVs

Because the tag is baked into the filename, you can query results with simple
grep / glob / pandas filters without parsing a separate manifest.

---

## Tag format by algorithm

### single-pass

```
single_orig{S}
```

- `S` = original sample index (0-based)

No branching -- one sample in, one sample out.

### best-of-n

```
bon_orig{S}_r{R}
```

- `S` = original sample index
- `R` = replica index (0 to replicas-1)

All replicas of the same sample share the prefix `bon_orig{S}`.

### beam-search

```
beam_orig{S}_bm{B}[-s{START}to{END}br{BR}]*
```

Components:

- `beam` -- algorithm prefix
- `orig{S}` -- original sample index
- `bm{B}` -- initial beam index (0 to beam_width-1)
- `-s{START}to{END}br{BR}` -- one segment per search step:
  - `START`, `END` = denoising step range (from `step_checkpoints`)
  - `BR` = branch index chosen/explored at this step

Example with `step_checkpoints: [0, 100, 200, 300, 400]`, beam_width=4, n_branch=4:

```
beam_orig0_bm2-s0to100br3-s100to200br0-s200to300br1-s300to400br2
|         |    |            |              |              |
|         |    |            |              |              step 4: denoised 300->400, branch 2
|         |    |            |              step 3: denoised 200->300, branch 1
|         |    |            step 2: denoised 100->200, branch 0
|         |    step 1: denoised 0->100, branch 3
|         initial beam replica 2
original sample 0
```

Samples captured at earlier steps (lookaheads) have fewer segments.  A
lookahead from step 2 looks like: `beam_orig0_bm2-s0to100br3-s100to200br0`.

### fk-steering

Same format as beam search, with prefix `fk`:

```
fk_orig{S}_bm{B}[-s{START}to{END}br{BR}]*
```

### mcts

```
mcts_orig{S}[-s{START}to{END}br{BR}]*
```

Lineage segments are appended each time the tree moves to its best child.
Lookahead (simulation) tags additionally include the simulation index:

```
mcts_orig{S}[-s{START}to{END}br{BR}]*-s{START}to{END}sim{SIM}br{BR}
```

- `sim{SIM}` = which simulation produced this sample

---

## Output size formulas

### beam-search / fk-steering (with keep_lookahead_samples: true)

```
Total PDBs = N * W * (B * S + 1)

N = nsamples               (batch size from dataloader)
W = beam_width             (beams kept per sample after top-k)
B = n_branch               (branches explored per beam per step)
S = len(step_checkpoints) - 1
```

Breakdown:

- Lookahead PDBs per step: `N * W * B` (all candidates, selected or not)
- Lookahead PDBs total: `N * W * B * S`
- Final PDBs: `N * W` (the surviving beams)

Example (N=4, W=4, B=4, checkpoints=[0,100,200,300,400], S=4):

```
Lookahead = 4 * 4 * 4 * 4 = 256
Final     = 4 * 4          =  16
Total     = 256 + 16       = 272
```

With `keep_lookahead_samples: false`, only the `N * W` final PDBs are saved.
With `reward_threshold` set, lookahead PDBs below the threshold are dropped.

### best-of-n

```
Total PDBs = N * replicas
```

### mcts

```
Total PDBs = N + (N * n_simulations * S)   [with keep_lookahead_samples: true]
Total PDBs = N                              [without]
```

---

## Querying search trajectories

All examples below assume you have a results CSV with a `metadata_tag` column
and PDB files whose names contain the tag.

### 1. Full journey of a winning sample

Given a final winner:

```
beam_orig0_bm0-s0to100br3-s100to200br0-s200to300br1-s300to400br2
```

Its **ancestors** are the lookaheads at each earlier step that share its lineage
prefix.  Each prefix represents the state that was selected to continue:

| Step | Prefix to search | What it finds |
|------|-----------------|---------------|
| 1 | `beam_orig0_bm0-s0to100br3` | The lookahead from step 1 that won |
| 2 | `beam_orig0_bm0-s0to100br3-s100to200br0` | Step 2 winner |
| 3 | `beam_orig0_bm0-s0to100br3-s100to200br0-s200to300br1` | Step 3 winner |
| 4 | Full tag | The final sample |

```bash
# Find the winner's journey (all ancestors + the final)
grep "beam_orig0_bm0-s0to100br3" results.csv
```

This returns one row per step -- the lookahead that was evaluated at that
checkpoint along this lineage.

### 2. Siblings: branches from the same parent (very similar structures)

At the last step, the winner's parent beam branched `n_branch` times.  These
siblings share the entire denoising trajectory up to step 300 and only differ in
the final 100 steps -- they are structurally very similar.

```bash
# All 4 siblings from the winner's parent at the last step
grep "beam_orig0_bm0-s0to100br3-s100to200br0-s200to300br1-s300to400br" results.csv
```

This matches `br0`, `br1`, `br2`, `br3` -- exactly the `n_branch` candidates
from this beam at the last step.

To get siblings at an earlier step (e.g., step 2):

```bash
grep "beam_orig0_bm0-s0to100br3-s100to200br" results.csv
```

### 3. All competitors in the top-k pool at a given step

At each step, ALL surviving beams branch and compete in the same top-k pool.
The winner came from beam `bm0`, but beams `bm1`, `bm2`, `bm3` also had
candidates.  To find everyone who competed at the last step for sample `orig0`:

```bash
# All candidates at step 300->400 for orig0 (across all beams)
grep "beam_orig0.*s300to400" results.csv
```

This returns `beam_width * n_branch` rows (e.g., 4 * 4 = 16) -- the full pool
that top-k selected from.

### 4. Track a beam across steps

To see how a specific beam evolved through the search (which branches it
took at each step):

```bash
# Everything from beam 2 of sample 0
grep "beam_orig0_bm2" results.csv
```

The rows returned show the full tree of branches explored from this beam.

### 5. Compare lookahead quality across steps

Lookahead samples have `sample_type=lookahead` in the CSV; final samples have
`sample_type=final`.  The `total_reward` column is the reward computed during
search.

```python
import pandas as pd

df = pd.read_csv("results.csv")

# Reward distribution at each step for orig0
for step_range in ["s0to100", "s100to200", "s200to300", "s300to400"]:
    step_df = df[df.metadata_tag.str.contains(step_range) &
                 df.metadata_tag.str.startswith("beam_orig0")]
    print(f"{step_range}: {len(step_df)} candidates, "
          f"reward range [{step_df.total_reward.min():.3f}, "
          f"{step_df.total_reward.max():.3f}]")
```

### 6. MCTS: trace the chosen path

For MCTS, the final tag encodes the tree-walk decisions:

```
mcts_orig0-s0to100br1-s100to200br0
```

All simulations (lookaheads) from a given checkpoint:

```bash
# All simulations at checkpoint 0->100 for sample 0
grep "mcts_orig0-s0to100sim" results.csv
```

The simulation that found the selected branch:

```bash
# Which simulations explored branch 1 at step 0->100?
grep "mcts_orig0-s0to100sim.*br1" results.csv
```

---

## Key properties of the tag scheme

- **Prefix = ancestry**: if tag A is a prefix of tag B, then A is an ancestor
  of B in the search tree.
- **Filename = provenance**: the PDB filename contains the full tag, so you
  can identify any file's lineage without opening a CSV.
- **Step values, not indices**: `s0to100` tells you the actual denoising range,
  not just "step 1".  This is important when comparing runs with different
  `step_checkpoints`.
- **Grep-friendly**: all queries above use simple substring/prefix matching.
  No parsing or JSON needed.
