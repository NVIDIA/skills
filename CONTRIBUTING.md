# Contributing to NVIDIA Agent Skills

This repository is a **catalog** — it lists NVIDIA-verified skills and links to the source repos where they live. Skills themselves are maintained by each product team in their own repos.

All participants are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Contributing to Skills

To contribute to an existing skill or propose a new one, use the contributing guidelines in the relevant product repo:

| Product | Contributing Guide |
|---------|-------------------|
| cuOpt | [NVIDIA/cuopt CONTRIBUTING.md](https://github.com/NVIDIA/cuopt/blob/main/CONTRIBUTING.md) |
| TensorRT-LLM | [NVIDIA/TensorRT-LLM CONTRIBUTING.md](https://github.com/NVIDIA/TensorRT-LLM/blob/main/CONTRIBUTING.md) |
| Nemotron Voice Agent | [NVIDIA-AI-Blueprints/nemotron-voice-agent CONTRIBUTING.md](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent/blob/main/CONTRIBUTING.md) |
| NeMo Gym | [NVIDIA-NeMo/Gym CONTRIBUTING.md](https://github.com/NVIDIA-NeMo/Gym/blob/main/CONTRIBUTING.md) |

---

## Contributing to This Catalog

For changes to this repo itself (README, adding a new product listing, fixing links):

1. Fork the repository.
2. Create a feature branch: `git checkout -b fix/description`
3. Make your changes.
4. Open a Pull Request with a clear description.

### Pull Request Checklist

- [ ] All commits are signed off (`git commit -s`) — see [Signing Your Work](#signing-your-work)
- [ ] No secrets, credentials, or API keys included
- [ ] Links verified and working

### Review Process

NVIDIA maintainers review all pull requests. Expect acknowledgement within a few business days. If your PR hasn't received a response after a week, feel free to leave a comment to bump it.

---

## Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129

    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.
  ```

  ```
    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or

    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or

    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.

    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.
  ```
