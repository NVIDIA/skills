## Description

Add `ClassName` prognostic model for BRIEF_DESCRIPTION.

Blocked by: OPTIONAL_PR_OR_ISSUE
Closes: OPTIONAL_ISSUE

### Model details

| Property | Value |
|---|---|
| **Architecture** | PyTorch / ONNX / JAX / other |
| **Time step** | Xh |
| **Input variables** | N variables (brief groups or link to implementation) |
| **Output variables** | N variables (brief groups or link to implementation) |
| **Spatial resolution** | X degree x Y degree (NxM grid) |
| **History required** | None / Xh / lead-time list |
| **Checkpoint source** | HuggingFace / NGC / S3 / other |
| **Checkpoint size** | XX MB / GB |
| **Checkpoint license** | LICENSE_NAME / Unknown |
| **Reference** | PAPER_OR_REPO_URL |
| **GitHub** | REPO_URL |

If the checkpoint or model-weight license is not clearly documented, write
`Unknown` and explain where you checked in the PR text or a follow-up comment.

### Dependencies added

| Package | Version | License | License URL | Reason |
|---|---|---|---|---|
| `PACKAGE_NAME` | `>=X.Y` | LICENSE_TYPE | [link](LICENSE_URL) | REASON |

Every model must have a dependency extra. If the extra is empty, write that
no third-party packages are added and name the empty model extra. For each new
dependency, check the package repository or package index, link directly to the
license, and flag any non-permissive license for maintainer review.

### Reference comparison

Reference comparison validated against the vanilla third-party implementation
using matched inputs and deterministic settings where applicable.

**Single step:**
- Max absolute difference: VALUE
- Mean absolute difference: VALUE
- Correlation: VALUE

**Multi-step (N steps / LEAD_TIME):**
- Step 1: max_abs=VALUE, corr=VALUE
- Step N: max_abs=VALUE, corr=VALUE

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
