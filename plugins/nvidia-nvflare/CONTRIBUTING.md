- [Introduction](#introduction)
- [The contribution process](#the-contribution-process)
  * [Preparing pull requests](#preparing-pull-requests)
    1. [Checking the coding style](#checking-the-coding-style)
    1. [Unit testing](#unit-testing)
    1. [Building the documentation](#building-the-documentation)
    1. [Signing your work](#signing-your-work)
    1. [Commit signature verification](#commit-signature-verification)
  * [Submitting pull requests](#submitting-pull-requests)
- [The code reviewing process (for the maintainers)](#the-code-reviewing-process)
  * [Reviewing pull requests](#reviewing-pull-requests)

## Introduction


Welcome to NVIDIA FLARE! We're excited you're here and want to contribute.
This documentation is intended for individuals and institutions interested in contributing to NVIDIA FLARE.
NVIDIA FLARE is an open-source project and, as such, its success relies on its community of contributors willing to keep improving it.
Your contribution will be a valued addition to the code base;
we simply ask that you read this page and understand our contribution process, whether you are a seasoned open-source contributor or whether you are a first-time contributor.

### Communicate with us

We are happy to talk with you about your needs for NVIDIA FLARE and your ideas for contributing to the project.
One way to do this is to create an issue discussing your thoughts. It might be that a very similar feature is under development or already exists, so an issue is a great starting point.

## The contribution process

_Pull request early_

We encourage you to create pull requests early.
It helps us track the contributions under development, whether they are ready to be merged or not. Change your pull request's title, to begin with `[WIP]` and/or [create a draft pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests#draft-pull-requests) until it is ready for formal review.


### Preparing pull requests
To ensure the code quality, NVIDIA FLARE relies on several linting tools:
([flake8 and its plugins](https://github.com/pycqa/flake8), [black](https://github.com/psf/black) and [isort](https://github.com/timothycrosley/isort))

This section highlights all the necessary preparation steps required before sending a pull request.
To collaborate efficiently, please read through this section and follow them.

* [Checking the coding style](#checking-the-coding-style)
* [Unit testing](#unit-testing)
* [Building documentation](#building-the-documentation)
* [Signing your work](#signing-your-work)


#### Checking the coding style
We check code style using flake8 and isort.
A bash script (`runtest.sh`) is provided to run all tests locally.
You can use `runtest.sh -f` to use black to fix your code style automatically as well.

License information: all source code files should start with this paragraph:
```
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

```

#### Unit testing
NVIDIA FLARE tests are located under test/.
The unit test file names follow the `[module_name]_test.py` pattern.

The bash script `runtest.sh` will run unit tests also.

#### Building the documentation

Use a dedicated documentation environment to preserve training dependencies.
From the repository root, create and activate a new environment, then build:

```bash
python3.10 -m venv .venv-docs
source .venv-docs/bin/activate
./build_doc.sh --html
```

The script installs the documentation dependencies. Linux x86_64 with Python 3.10
uses the same pinned CPU framework environment as Read the Docs; other supported
environments use the documentation extra. Keep this environment separate from
the development or training environment.

Open `docs/_build/index.html` to read the generated HTML. To clean generated files:

```bash
./build_doc.sh --clean
```

The production dependency pins are in `docs/constraints.txt`. Update them in a
fresh documentation environment and verify the full HTML and Markdown build
before committing new versions.

#### Signing your work
NVIDIA FLARE enforces the [Developer Certificate of Origin](https://developercertificate.org/) (DCO) on all pull requests.

For a detailed guide on signing commits, please see [Signing commits](https://docs.github.com/en/authentication/managing-commit-signature-verification/signing-commits) from GitHub.

#### Commit signature verification
NVIDIA FLARE enforces commit signature verification, a security feature provided by GitHub. Developers are required to setup GPG keys as described in [Commit Signature Verification](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification#gpg-commit-signature-verification).

Full text of the DCO:
```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

### Submitting pull requests
All code changes to the main branch must be done via [pull requests](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests).
1. Create a new ticket or take a known ticket from [the issue list][NVIDIA FLARE issue list].
2. Check if there's already a branch dedicated to the task.
3. If the task has not been taken, [create a new branch in your fork](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork)
of the codebase.
Ideally, the new branch should be based on the latest `main` branch.
4. Make changes to the branch ([use detailed commit messages if possible](https://chris.beams.io/posts/git-commit/)).
5. Make sure that new tests cover the changes and the changed codebase [passes all tests locally](#unit-testing).
6. [Create a new pull request](https://help.github.com/en/desktop/contributing-to-projects/creating-a-pull-request) from the task branch to the main branch, with detailed descriptions of the purpose of this pull request.
7. Check [the CI/CD status of the pull request][github ci], make sure all CI/CD tests passed.
8. Assign 2 reviewers. One of the reviewers must be a code owner for this section of code.
9. Wait for reviews; if there are reviews, make point-to-point responses, make further code changes if needed.
10. If there are conflicts between the pull request branch and the main branch, pull the changes from the main and resolve the conflicts locally.
11. Reviewer and contributor may have discussions back and forth until all comments addressed. All conversations must be resolved for PR to pass.
12. Wait for the pull request to be merged.

## The code reviewing process


### Reviewing pull requests
All code review comments should be specific, constructive, and actionable.
1. Check [the CI/CD status of the pull request][github ci], make sure all CI/CD tests passed before reviewing (contact the branch owner if needed).
2. Read carefully the descriptions of the pull request and the files changed, write comments if needed.
3. Make in-line comments to specific code segments, [request for changes](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-request-reviews) if needed.
4. Review any further code changes until all comments addressed by the contributors.
5. Merge the pull request to the main branch.
6. Close the corresponding task ticket on [the issue list][NVIDIA FLARE issue list].

[github ci]: https://github.com/NVIDIA/NVFlare/actions
[NVIDIA FLARE issue list]: https://github.com/NVIDIA/NVFlare/issues
