# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Initialize process-wide settings needed before FastGen submodules import."""

import logging
from pathlib import Path

from dotenv import load_dotenv

# Many modules read configuration from the environment at import time. Load the repo-root .env
# first when present (anchored to the package, not the cwd); real env vars still win.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class _TorchAOOptionalExtensionFilter(logging.Filter):
    """Hide known failures for optional TorchAO extensions with mismatched ABIs."""

    _OPTIONAL_EXTENSIONS = ("/torchao/_C_mxfp8", "/torchao/_C_cutlass_90a")

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (
            record.name == "torchao"
            and message.startswith("Failed to load ")
            and any(extension in message for extension in self._OPTIONAL_EXTENSIONS)
        )


# TorchAO logs these while importing, before FastGen's normal logging setup runs.
logging.getLogger("torchao").addFilter(_TorchAOOptionalExtensionFilter())
