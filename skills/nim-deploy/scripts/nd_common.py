#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One import for the shared helpers of the nim-deploy scripts: `import nd_common as nd`.

nd_state holds the run state, the transitions and the JSON output contract; nd_render holds the
YAML writer, strict input parsing and the open/close steps of a render. Stdlib only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nd_state
from nd_render import *  # noqa: F403
from nd_state import *  # noqa: F403

if __name__ == "__main__":
    nd_state.emit(True, "library", "done", {"note": "import this module; it has no command line"})
