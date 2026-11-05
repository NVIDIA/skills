# Draft pull request (synthetic example prepared for a review exercise)

- Target repository: NVIDIA/TensorRT-LLM, base branch `main`
- Head branch: `jdoe/short-seq-mha` on the author's personal fork (git remote `jdoe`)
- Tracking ticket: JIRA TRTLLM-6021
- Author is an NVIDIA developer.

## Proposed PR title

    Short seq MHA for MLA

## Proposed PR description

    Adds short MHA path.

## Commits on the branch (oldest first)

Output of
`git log --reverse --format='%h | %an <%ae> | %s | trailers: %(trailers:only,unfold)' main..HEAD`:

    3c1d2a0 | Jane Doe <dev-placeholder@nvidia.com> | Add short-seq MHA path for DSA MLA context attention | trailers:
    7a9e1f4 | Jane Doe <dev-placeholder@nvidia.com> | fix | trailers:
    b52c0d9 | Jane Doe <dev-placeholder@nvidia.com> | fix the fix for cached tokens | trailers:
    e0f7781 | Jane Doe <dev-placeholder@nvidia.com> | Add custom RoPE helper to rotary_embedding.py | trailers:
    91d4c3e | Jane Doe <dev-placeholder@nvidia.com> | Revert "Add custom RoPE helper to rotary_embedding.py" | trailers:
    c8e2b10 | Jane Doe <dev-placeholder@nvidia.com> | Reformat attention.py (377 lines, formatting only) | trailers:
    f4a6d27 | Jane Doe <dev-placeholder@nvidia.com> | Add tests | trailers:

## Diff stat (main...HEAD)

    tensorrt_llm/_torch/attention/attention.py         | 214 +++++++++++++-----
    tests/unittest/_torch/attention/test_short_mha.py  | 665 +++++++++++++++++++++++
    2 files changed, 820 insertions(+), 59 deletions(-)
