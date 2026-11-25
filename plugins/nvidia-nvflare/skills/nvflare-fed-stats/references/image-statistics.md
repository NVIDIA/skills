# Image Dataset Federated Statistics

The image path shares the tabular path's recipe, privacy filters, automatic
execution, and validation ladder; this reference holds everything that
differs. Generate the client from `assets/image_stats_client.py`, not from
the tabular template.

## Detection

`nvflare agent inspect data <path> --format json` classifies image data:
`dataset.modality: image` with a `dataset` block carrying the extension
census, per-site file counts, per-site `image_formats` (pick the loader
from these), and a sampled `pixel_depth` (e.g. `uint8`; stays null for
DICOM/NIfTI, where depth must come from the format-specific loader). For
`.dcm`/`.nii` sites the generated client must extend the template's
discovery extensions and swap `_load_image` for the preflighted loader.
Select the image/tabular path from `dataset.modality` rather than re-deriving
it; `counts_approximate:
true` means the walk hit its file limit — verify site counts directly
before bin-cap decisions. Datalist-JSON layouts may return `dataset: null`;
read the datalist as declared layout evidence in that case. Companion tabular
metadata (a `labels.csv` beside the scans) keeps
the dataset `image` and appears per site as `tabular_companions`: mention
it in the report, never treat it as a statistics target — but offer a
follow-up tabular statistics run over the labels (label shift across
sites is a classic QA question). Genuinely mixed
data — an image-only site among tabular sites, or materially both
modalities — reports `modality: mixed`: read the dataset block, report the
split, and run the two modalities as two separate jobs.

## Supported Statistics (Image)

The image feature is pixel `intensity`; the meaningful statistic set is:

```python
statistic_configs = {
    "count": {},          # number of image files discovered at the site
    "failure_count": {},  # unreadable/corrupt files; MUST be configured to appear
    "histogram": {"*": {"bins": 20, "range": [0, 256]}},
}
```

- `count` is the number of discovered image files, not pixels; it includes
  unreadable files, which `failure_count` reports.
- `failure_count` is correct from round 1: `initialize()` verifies every
  file with the same full-decode operation the histogram uses, so
  unreadable files (broken headers AND broken pixel data) are counted
  before the server's first collection and the `Global` row accumulates
  them. A file failing AFTER verification (mutated or transiently
  unreadable between rounds) is added to `failure_count` during the
  histogram pass, the histogram excludes that file, and the returned
  second-round payload lets the standard min-count/bin-cap filters screen
  the output against the honest effective count. Do not raise from the
  statistic method for this condition: the statistics task handler catches
  per-statistic exceptions and can otherwise return partial output. This
  product-side exception/filter ordering is tracked with NVFlare issue
  #4876. Report the late failure as a data-quality finding with rerun
  guidance. Very large datasets may trade the double-decode for
  round-2-only discovery; say so in the report.
- mean/sum/stddev/var/quantile/min/max over image sets are not supported by
  this skill version: report them as unsupported for image data rather than
  improvising pixel-pooled implementations.

## Histogram Range And Bins

- The intensity range comes from the pixel dtype/bit depth, which is
  legitimate static evidence: `[0, 256]` for 8-bit, `[0, 65536]` for
  16-bit, `[0, 1]` for normalized floats. Use the dataset block's sampled
  `pixel_depth` and state the choice; a user/README declaration overrides.
- The bin-cap cleanser compares bins to the site's *image count* (not pixel
  count): bins must be under round(10% of effective count), so 20 bins
  first passes at 206 readable images per site. Size the default bin count
  using the dataset block's `site["image_files"]` — NOT `data_files`,
  which also counts companion label files — against the smallest site,
  and state the choice, exactly as in the tabular path.

## Loaders And Dependencies

- pandas is NOT an image-path dependency: probing it on an image run only
  creates dependency noise. The image path needs numpy plus a loader.
- PIL (Pillow) covers PNG/JPEG/BMP/TIFF and is the template default, with
  grayscale conversion before histogramming — state that policy in the
  report; per-channel statistics are not supported in this version.
- DICOM requires `pydicom` (or MONAI's ITK reader); NIfTI requires
  `nibabel`. Preflight the import for the format actually present before
  generating the job, exactly like the fastdigest rule: on failure, fail
  closed with the product error, never silently skip files. Any format
  whose loader yields a pixel array is supported — the loader is the only
  gate.
- CT DICOM correctness: stored values become Hounsfield Units only after
  `RescaleSlope * value + RescaleIntercept`; apply the rescale in the
  loader and use a declared HU range (e.g. `[-1024, 3072]`) — never
  histogram raw stored values as calibrated intensities.
- Count semantics for medical formats: `count` is files, which for DICOM
  series means slices (not studies/patients) and for NIfTI means whole
  volumes; say which in the report. Cross-site intensity histograms also
  confound scanner/protocol calibration with population differences —
  include that caveat alongside the case-mix one.

## Validation Specifics

The ladder is unchanged; the statistics rungs adapt as below. Runtime
facts (where output lands, execute semantics) come from
`stats-job-validation.md` — load it instead of reading NVFLARE library
source; the output JSON is written under
`<workspace>/<job-name>/server/simulate_job/<stats_output_path>`.

The statistics rungs:

- **Completeness** — the output hierarchy is
  `{"intensity": {statistic: {site: {dataset: value}}}}` with every site
  plus `Global` under each configured statistic; histogram leaves are
  lists of `[low, high, count]` triples, counts are plain integers —
  probe the actual JSON shapes before parsing (ephemeral commands only).
- **Parity (harness-owned, not performed by the skill)** — for offline
  verification: recompute one site's histogram independently (same
  loader, same grayscale conversion, same bins/range) and compare bin
  counts exactly; `Global` histogram bins equal the element-wise sum of
  the site bins, and `Global` count the sum of site counts.
- **Failure reporting** — nonzero `failure_count` at a site is a data-quality
  finding to report (corrupt or unreadable files), not a run failure; the
  affected images are simply absent from that site's histogram.

Report shape follows the tabular path (aggregates only — bin counts and
file counts, never pixel data or file-name listings that could identify
patients), with the same case-mix caveat for `Global`.
