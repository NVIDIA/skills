## Reference Comparison Validation

**Model:** `ClassName`
**Comparison date:** YYYY-MM-DDTHH:MM:SS
**Environment:** GPU / CPU validation run (do not include machine names, hostnames, absolute paths, or device inventory)

### Results Summary

| Metric | Step 1 (+Xh) | Step N (+Yh) |
|---|---:|---:|
| Max absolute difference | VALUE | VALUE |
| Mean absolute difference | VALUE | VALUE |
| Correlation | VALUE | VALUE |
| Finite fraction | VALUE | VALUE |

**Key findings:**
- BULLET_SINGLE_STEP_AGREEMENT
- BULLET_MULTISTEP_OR_AUTOREGRESSIVE_BEHAVIOR
- BULLET_SPATIAL_PATTERN_QUALITY

### Reference Plots

Do not upload or attach images from the automation. Leave placeholders for the
PR author to upload images manually in the browser.

Single-step sanity plots:

> TODO: Upload `SINGLE_STEP_PLOT_NAME_1` here.
>
> TODO: Upload `SINGLE_STEP_PLOT_NAME_2` here.

Multi-step vanilla-vs-Earth2Studio comparison plots. For each selected variable,
the plot should have forecast lead times as columns and these rows: Earth2Studio
wrapper output on top, vanilla reference output in the middle, and relative
error on the bottom. Use a small denominator clamp such as `max(abs(vanilla), 1e-6)`.

> TODO: Upload `REFERENCE_COMPARE_TIMESERIES_VARIABLE_1.png` here.
>
> TODO: Upload `REFERENCE_COMPARE_TIMESERIES_VARIABLE_2.png` here.
>
> TODO: Upload `REFERENCE_COMPARE_TIMESERIES_VARIABLE_3.png` here.

---

### Validation Scripts

The validation scripts are for review only and must not be committed. Include
copy-pasteable Python scripts here, but remove machine names, absolute paths,
cache paths, hostnames, and environment-specific details.

<details>
<summary>Vanilla reference script</summary>

```python
# Before running this vanilla reference script, clone the original U-CAST
# repository (Rose-STL-Lab/u-cast), then set UCAST_REFERENCE_REPO to that
# checkout or run from a directory containing ./u-cast.
PASTE THE FULL WORKING VANILLA SCRIPT HERE.
```

</details>

<details>
<summary>Earth2Studio reference script</summary>

```python
PASTE THE FULL WORKING E2S SCRIPT HERE.
```

</details>

<details>
<summary>Comparison script</summary>

```python
PASTE THE FULL WORKING COMPARISON SCRIPT HERE.
```

</details>
