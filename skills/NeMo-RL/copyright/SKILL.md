---
name: copyright
description: NVIDIA copyright header requirements for NeMo-RL. Covers which files need headers and the exact header text.
when_to_use: Adding a new file; checking copyright headers; 'missing copyright', 'NVIDIA header', 'license header', 'is a header required here', during code review.
---

# NVIDIA Copyright Header

Add the following NVIDIA copyright header to all Python files and shell scripts. The header should include the current year. Exclude test files (files under `tests/` or test-only scripts).

The header must appear at the top of the file. **Always use the current year** (not a hardcoded year):

```python
# Copyright (c) <CURRENT_YEAR>, NVIDIA CORPORATION.  All rights reserved.
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
