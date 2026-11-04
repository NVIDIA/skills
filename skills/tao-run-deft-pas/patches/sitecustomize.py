# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allow numpy checkpoint scalars under torch's weights-only loader.

TAO CLIP optimizer checkpoints may contain numpy scalar/dtype objects.  Torch
2.6+ rejects those objects unless their concrete dtype classes are allowlisted.
The platform action mounts or injects this module through ``PYTHONPATH``; keep this behavior
for the skill until the container includes the fix.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from importlib.abc import Loader, MetaPathFinder


def _register_safe_globals(torch_module):
    try:
        import numpy as np

        serialization = importlib.import_module("torch.serialization")
        safe_globals = [np.dtype, np.ndarray]
        dtypes_module = getattr(np, "dtypes", None)
        if dtypes_module is not None:
            for name in dir(dtypes_module):
                candidate = getattr(dtypes_module, name)
                if isinstance(candidate, type) and issubclass(candidate, np.dtype):
                    safe_globals.append(candidate)
        try:
            from numpy._core.multiarray import scalar as np_scalar
        except ImportError:
            from numpy.core.multiarray import scalar as np_scalar
        safe_globals.append(np_scalar)
        serialization.add_safe_globals(safe_globals)
    except Exception:
        # A compatibility patch must never make plain interpreter startup fail.
        pass


class _PostImportLoader(Loader):
    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        _register_safe_globals(module)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _TorchImportHook(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "torch":
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        except Exception:
            return None
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PostImportLoader(spec.loader)
        return spec


if "torch" in sys.modules:
    _register_safe_globals(sys.modules["torch"])
else:
    sys.meta_path.insert(0, _TorchImportHook())
