# Validation Guide

> Reference for post-implementation validation, PR submission, and automated review.
> Load this after implementation and tests are complete.

## Table of Contents

- [Run Tests](#run-tests)
- [Validate Variables](#validate-variables)
- [Sanity-Check Plots](#sanity-check-plots)
- [Branch, Commit and Open PR](#branch-commit-and-open-pr)
- [Automated Code Review (Greptile)](#automated-code-review-greptile)

---

## Run Tests

### Run the new test file

```bash
uv run python -m pytest test/data/test_<filename>.py -v --timeout=60
```

All tests must pass (or `xfail` for network tests).

### Run coverage with `--slow`

The source file must achieve **at least 90% line coverage**:

```bash
uv run python -m pytest test/data/test_<filename>.py -v \
    --slow --timeout=300 \
    --cov=earth2studio/data/<filename> \
    --cov-report=term-missing \
    --cov-fail-under=90
```

If below 90%, add tests for:
- Error handling branches
- Edge cases in parsing
- Cache property paths (`cache=True` vs `cache=False`)
- `resolve_fields` with different input types

### Full data test suite (optional)

```bash
make pytest TOX_ENV=test-data
```

---

## Validate Variables

**Every variable in the lexicon must be validated against real data.**

Run a validation script (do NOT commit):

```python
"""Variable validation for <SourceName>."""
from datetime import datetime
from earth2studio.data import SourceName
from earth2studio.lexicon import SourceNameLexicon

ds = SourceName(cache=True)
time = ...  # pick a recent valid time
all_vars = list(SourceNameLexicon.VOCAB.keys())
df = ds(time, all_vars)

print(f"{'Variable':<16} {'Obs Count':>10} {'Valid %':>8} {'Min':>10} {'Max':>10}")
print("-" * 60)
for var in sorted(all_vars):
    sub = df[df["variable"] == var]
    n_total = len(sub)
    n_valid = sub["observation"].notna().sum()
    pct = (n_valid / n_total * 100) if n_total > 0 else 0
    vmin = sub["observation"].min() if n_valid > 0 else float("nan")
    vmax = sub["observation"].max() if n_valid > 0 else float("nan")
    flag = " *** REMOVE" if pct < 10 else ""
    print(f"{var:<16} {n_total:>10} {pct:>7.1f}% {vmin:>10.2f} {vmax:>10.2f}{flag}")
```

**Actions:**
- Variables with **< 10% valid data**: Remove from lexicon VOCAB, E2STUDIO_VOCAB, and document
- Variables with **10-50% valid data**: Keep but document low coverage reason
- After removing, re-run `make lint` and tests

### Summarize to user

Present a summary table covering:
1. All valid variables — name, description, observation count, value range
2. Removed variables — name, reason
3. Valid time range
4. Typical data density

---

## Sanity-Check Plots

Create a standalone script (do NOT commit).

### Gridded template (DataSource / ForecastSource)

```python
"""Sanity-check plot for <SourceName>."""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from earth2studio.data import SourceName

ds = SourceName(cache=False)
time = ...
variables = ["t2m", "msl", "u10m"]
data = ds(time, variables)

fig, axes = plt.subplots(
    1, len(variables), figsize=(6 * len(variables), 5),
    subplot_kw={"projection": ccrs.PlateCarree()},
)
if len(variables) == 1:
    axes = [axes]

for ax, var in zip(axes, variables):
    arr = data.sel(variable=var).isel(time=0)
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    im = ax.pcolormesh(
        arr.lon, arr.lat, arr.values,
        cmap="turbo", transform=ccrs.PlateCarree(),
    )
    ax.set_title(f"{var}")
    plt.colorbar(im, ax=ax, shrink=0.6, orientation="horizontal", pad=0.05)

plt.suptitle(f"<SourceName> — {time}", y=1.02)
plt.tight_layout()
plt.savefig("sanity_check_<source_name>.png", dpi=150, bbox_inches="tight")
```

### Sparse template (DataFrameSource / ForecastFrameSource)

```python
"""Sanity-check plot for <SourceName>."""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from earth2studio.data import SourceName

ds = SourceName(cache=False)
time = ...
variables = ["var1", "var2"]
df = ds(time, variables)

df["lon_plt"] = df["lon"].where(df["lon"] <= 180, df["lon"] - 360)

fig, axes = plt.subplots(
    1, len(variables), figsize=(8 * len(variables), 5),
    subplot_kw={"projection": ccrs.Robinson()},
)
if len(variables) == 1:
    axes = [axes]

for ax, var in zip(axes, variables):
    subset = df[df["variable"] == var]
    obs = subset["observation"].values
    vmin, vmax = np.percentile(obs[np.isfinite(obs)], [2, 98])
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    sc = ax.scatter(
        subset["lon_plt"], subset["lat"], c=obs, s=2, cmap="turbo",
        alpha=0.8, vmin=vmin, vmax=vmax, edgecolors="none",
        transform=ccrs.PlateCarree(),
    )
    ax.set_title(f"{var} ({len(subset)} obs)")
    plt.colorbar(sc, ax=ax, shrink=0.6, orientation="horizontal", pad=0.05)

plt.tight_layout()
plt.savefig("sanity_check_<source_name>.png", dpi=150, bbox_inches="tight")
```

### User confirmation required

Tell the user the absolute path to the plot and ask them to visually confirm:
1. Data points visible (not blank/empty)
2. Geographic coverage matches expectations
3. Colorbar values in physically reasonable ranges
4. No obvious artifacts

**Do not proceed until user confirms.**

---

## Branch, Commit and Open PR

### Create branch and commit

```bash
git checkout -b feat/data-source-<name>
git add earth2studio/data/<filename>.py \
        earth2studio/data/__init__.py \
        earth2studio/lexicon/<filename>.py \
        earth2studio/lexicon/__init__.py \
        earth2studio/lexicon/base.py \
        test/data/test_<filename>.py \
        docs/modules/datasources_*.rst \
        pyproject.toml \
        CHANGELOG.md
git commit -m "feat: add <SourceName> data source

Add <SourceName> <source_type> for <brief description>.
Includes lexicon, unit tests, and documentation."
```

Do NOT add sanity-check script or images.

### Push to fork

```bash
git remote -v  # identify fork remote
git push -u <fork-remote> feat/data-source-<name>
```

### Open PR (fork → NVIDIA/earth2studio)

```bash
gh pr create \
  --repo NVIDIA/earth2studio \
  --base main \
  --head <fork-owner>:feat/data-source-<name> \
  --title "feat: add <SourceName> data source" \
  --body "..."
```

### PR body template

**You MUST include all sections below.** The PR body is the primary record of
data licensing and dependency changes for legal review.

````markdown
## Description

Add `<ClassName>` <source_type> for <brief description>.

### Data source details

| Property | Value |
|---|---|
| **Source type** | DataSource / ForecastSource / DataFrameSource / ForecastFrameSource |
| **Remote store** | <URL> |
| **Format** | GRIB2 / NetCDF / Zarr / etc. |
| **Spatial resolution** | X deg x Y deg (or "Point observations" for sparse) |
| **Temporal resolution** | Hourly / 6-hourly / daily |
| **Date range** | YYYY-MM-DD to present |
| **Authentication** | Anonymous / API key |

### Data licensing

> **License**: <Name> (e.g., Public Domain, CC-BY-4.0, Apache-2.0)
> **URL**: <Link to license or data policy page>
>
> <Brief summary of permissions/restrictions>

### Dependencies added

<!-- If no new dependencies, write: "No new dependencies required. Uses existing `<pkg1>` and `<pkg2>`." -->

| Package | Version | License | License URL | Reason |
|---|---|---|---|---|
| `<pkg>` | `>=X.Y` | <License> | [link](<URL>) | <reason> |

## Checklist

- [x] New or existing tests cover these changes.
- [x] The documentation is up to date.
- [x] The CHANGELOG.md is up to date.
- [ ] Assess and address Greptile feedback.
````

### Post sanity-check as PR comment

Post immediately after creating PR (before Greptile review). Use `gh pr comment`:

```bash
gh pr comment <PR_NUMBER> --repo NVIDIA/earth2studio --body "..."
```

**Required content:**

1. **Variable coverage table** — name, obs count, value range, unit
2. **Data validation summary** — regions/stations, time range, key statistics
3. **Key findings** — physically reasonable values, unit conversions verified
4. **Full validation script** in `<details>` block
5. **Image placeholder** for user to drag-and-drop:

```markdown
### Sanity-Check Plot

<!-- Drag and drop sanity-check image here -->
```

---

## Automated Code Review (Greptile)

### Wait for review

Poll every 30s for up to 5 minutes:

```bash
for i in $(seq 1 10); do
  REVIEW_ID=$(gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/reviews \
    --jq '.[] | select(.user.login == "greptile-apps[bot]") | .id' 2>/dev/null)
  if [ -n "$REVIEW_ID" ]; then break; fi
  sleep 30
done
```

### Categorize feedback

| Category | Default action |
|---|---|
| Bug / correctness | Fix |
| Style / convention | Fix if valid |
| Performance | Evaluate |
| Documentation | Fix |
| Suggestion | User decides |
| False positive | Dismiss |

### Present to user

Show summary table with file, line, category, summary, and proposed action.
Ask user to confirm which comments to address.

### Implement fixes

1. Make code changes
2. Run `make format && make lint`
3. Run tests
4. Commit: `fix: address code review feedback (Greptile)`

### Respond to comments

**Fixed:**
```bash
gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/comments/<COMMENT_ID>/replies \
  -f body="Fixed in <commit_sha>. <description>"
```

**Dismissed:**
```bash
gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/comments/<COMMENT_ID>/replies \
  -f body="Won't fix — <justification>"
```

### Push and report

```bash
git push origin <branch>
```

Report to user: comments fixed, dismissed, and any remaining open threads.
