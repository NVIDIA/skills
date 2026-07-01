## Reference Comparison Validation

**Model:** `ClassName`
**Comparison date:** YYYY-MM-DDTHH:MM:SS
**Environment:** GPU / CPU validation run (do not include machine names, hostnames, absolute paths, cache paths, or device inventory)

### Results Summary

| Metric | Value |
|---|---:|
| Max absolute difference | VALUE |
| Mean absolute difference | VALUE |
| Correlation | VALUE |
| Finite fraction | VALUE |
| Output shape | VALUE |
| Coordinate order | VALUE |

For generative diagnostics:

| Metric | Value |
|---|---:|
| Seed(s) | VALUE |
| Number of samples | VALUE |
| Sample max absolute difference | VALUE |
| Sample correlation or distribution check | VALUE |

**Key findings:**
- BULLET_SINGLE_STEP_AGREEMENT
- BULLET_COORDINATE_AND_SHAPE_VALIDATION
- BULLET_GENERATIVE_OR_STOCHASTIC_BEHAVIOR_IF_APPLICABLE
- BULLET_SPATIAL_PATTERN_QUALITY

### Reference Plots

Do not upload or attach images from automation. Leave placeholders for the PR
author to upload images manually in the browser.

Single-step vanilla-vs-Earth2Studio comparison plots. For each selected variable,
the plot should show Earth2Studio output, vanilla reference output, and relative
error. Use a small denominator clamp such as `max(abs(vanilla), 1e-6)`.

> TODO: Upload `REFERENCE_COMPARE_VARIABLE_1.png` here.
>
> TODO: Upload `REFERENCE_COMPARE_VARIABLE_2.png` here.
>
> TODO: Upload `REFERENCE_SANITY_VARIABLE_1.png` here.

For generative diagnostics, include sample panels or seeded sample comparisons:

> TODO: Upload `GENERATIVE_SAMPLE_COMPARISON.png` here.

---

### Validation Scripts

The validation scripts are for review only and must not be committed. Include
copy-pasteable Python scripts here, but remove machine names, absolute paths,
cache paths, hostnames, and environment-specific details.

<details>
<summary>Vanilla reference script</summary>

```python
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

<details>
<summary>Sanity plot script</summary>

```python
PASTE THE FULL WORKING SANITY SCRIPT HERE.
```

</details>
