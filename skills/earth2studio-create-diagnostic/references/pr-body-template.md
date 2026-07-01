## Description

Add `ClassName` diagnostic model for BRIEF_DESCRIPTION.

Blocked by: OPTIONAL_PR_OR_ISSUE
Closes: OPTIONAL_ISSUE

### Diagnostic details

| Property | Value |
|---|---|
| **Diagnostic type** | Simple derived / AutoModel / Generative |
| **Framework** | PyTorch / ONNX / JAX / other |
| **Input variables** | N variables (brief groups or link to implementation) |
| **Output variables** | N variables (brief groups or link to implementation) |
| **Input spatial resolution** | X degree x Y degree (NxM grid) |
| **Output spatial resolution** | Same as input / X degree x Y degree (NxM grid) |
| **Samples** | Not applicable / N samples with seed support |
| **Checkpoint source** | None / HuggingFace / NGC / S3 / other |
| **Checkpoint size** | Not applicable / XX MB / GB |
| **Checkpoint license** | Not applicable / LICENSE_NAME / Unknown |
| **Reference** | PAPER_OR_REPO_URL |
| **GitHub** | REPO_URL |

If the checkpoint or model-weight license is not clearly documented, write
`Unknown` and explain where you checked in the PR text or a follow-up comment.

### Dependencies added

| Package | Version | License | License URL | Reason |
|---|---|---|---|---|
| `PACKAGE_NAME` | `>=X.Y` | LICENSE_TYPE | [link](LICENSE_URL) | REASON |

For simple derived diagnostics with no extra, state that no dependency extra is
needed. For AutoModel or generative diagnostics, name the model extra even when
it is empty. For each new dependency, check the package repository or package
index, link directly to the license, and flag any non-permissive license for
maintainer review.

### Reference comparison

Reference comparison validated against the vanilla third-party implementation
using matched inputs and deterministic settings where applicable.

**Single step:**
- Max absolute difference: VALUE
- Mean absolute difference: VALUE
- Correlation: VALUE
- Finite fraction: VALUE

**Generative diagnostics, if applicable:**
- Seed(s): VALUE
- Number of samples: VALUE
- Sample agreement or distribution summary: VALUE

### Validation

See the reference-comparison validation comment below. Plot placeholders are
included for manual image upload by the PR author.

## Checklist

- [x] I am familiar with the [Contributing Guidelines][contrib].
- [x] New or existing tests cover these changes.
- [x] The documentation is up to date with these changes.
- [x] The [CHANGELOG.md][changelog] is up to date with these changes.
- [ ] An [issue][issues] is linked to this pull request.
- [ ] Assess and address Greptile feedback (AI code review bot).

[contrib]: https://github.com/NVIDIA/earth2studio/blob/main/CONTRIBUTING.md
[changelog]: https://github.com/NVIDIA/earth2studio/blob/main/CHANGELOG.md
[issues]: https://github.com/NVIDIA/earth2studio/issues
