# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastGen module lifecycle: construct, prepare once, then train and checkpoint.

``FastGenModule.prepare_module`` prepares a root and every FastGen module registered below it, exactly
once. The code that builds a module prepares it, e.g. ``FastGenModel.init_preprocessors`` for
preprocessors, and later users share the prepared instance. Register all children before preparing
their root: children attached later stay unprepared, and ``FastGenModel.validate_modules`` rejects them.

Preparation and checkpointing are collective: every rank makes the same calls in the same order,
including preparation without a device. A failure after preparation has started changing state is not
rolled back, and nothing blocks a retry, so discard the modules and construct fresh ones. Deep copies
keep the preparation marker and recorded policy: copy before preparation, or load weights into a fresh
instance. Copying a live FSDP tree is unsupported.

Tied parameters and shared submodules within one root are supported through preparation, DDP and FSDP
training, and checkpoint resume. FSDP rejects shared submodules holding parameters or persistent buffers
when any rank is meta-initialized, and parameters shared across nested FastGen modules with differing
precision policies unless the root defines its own ``fully_shard`` layout.
"""

from typing import Dict, Iterator, Optional, Set, Tuple, TYPE_CHECKING

import torch

from fastgen.utils.distributed.collectives import raise_across_ranks, validate_object_across_ranks
from fastgen.utils.distributed.module import move_and_sync_module
from fastgen.utils.module import (
    cast_to_storage_dtypes,
    has_meta_state,
    reject_complex_parameters,
    reject_extra_state,
    reject_inference_state,
    reject_unsafe_placement,
    resolve_storage_dtypes,
)
import fastgen.utils.logging_utils as logger

if TYPE_CHECKING:
    from fastgen.utils.distributed.fsdp import FSDPContext


class FastGenModule(torch.nn.Module):
    """Base class owning a module's precision, mode, gradient, and preparation policy.

    * ``precision_model``: dtype of resolved conditioning, nominal execution without autocast, and
      frozen storage (``None`` = no casts). ``cast_to_model_precision`` casts primary neural inputs to
      it, promoted with an active autocast dtype, while uncast inputs such as conditioning must
      already be in it; dtypes derived inside the forward follow the cast primary input, not the
      conditioning. Active AMP takes precedence for eligible operators, and a
      strict precision region must own its local autocast policy. ``precision_opt``: storage dtype
      (``precision_model`` unless given). Network outputs keep the input state's dtype; preprocessors,
      discriminators, and rewards return their native compute dtype.
    * ``device``: the configured execution device shared by one physical module root and all of its
      registered children, not the momentary location of every parameter (FSDP shards may be meta or
      CPU-offloaded). ``prepare_module`` is its only exposed writer and places or shards the complete
      root; a plain ``.to(...)`` never redefines it. Callers remain responsible for placing inputs on
      that device. ``None`` = no target (nothing is moved, created tensors use the default device,
      and the caller or FSDP context owns placement); also the value before preparation.
    * ``prepare_module`` is the sole exposed lifecycle entry point. Its private ``_set_precision``
      helper lets every nested FastGen module resolve its own policy and cast only the state it owns.
      ``_set_device`` records one target across the registered tree; ``prepare_module`` then places
      or shards the complete root.
      Subclasses override the ``precision_model`` property for the state they own (their inputs are
      construction-time) and ``_fixed_precision_overrides`` for owned tensors; ``precision_opt`` and
      ``device`` are root-wide and cannot be overridden.
    * Preprocessors are wrappers around third-party modules, kept in a plain dict on their host
      network and prepared frozen by the model lifecycle. ``encode``/``decode`` expect
      caller-colocated inputs, cast them to the consuming module's precision, and route neural work
      through ``self(...)`` (an ``operation`` argument selects the entry point); callers cast the
      results to the precision they need.
    """

    _precision_model: Optional[torch.dtype] = None
    _precision_opt: Optional[torch.dtype] = None
    _device: Optional[torch.device] = None
    _is_prepared: bool = False
    # Mark selected logical roots after complete FSDP preparation.
    # Custom layouts often shard an inner transformer/UNet instead of making this object an
    # FSDPModule, so physical wrapper type is not an equivalent marker.
    _is_fsdp: bool = False

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Storage dtype and device are root-wide policies; only ``precision_model`` may be pinned.
        for name in ("precision_opt", "device"):
            if name in cls.__dict__:
                raise TypeError(f"{cls.__name__} overrides {name}; only precision_model may be overridden.")

    @property
    def is_prepared(self) -> bool:
        """Whether this module completed preparation, directly or through its registered root."""
        return self._is_prepared

    @property
    def precision_model(self) -> Optional[torch.dtype]:
        """Dtype of resolved conditioning and frozen storage; subclasses override to pin a module."""
        return self._precision_model

    def cast_to_model_precision(self, x: torch.Tensor) -> torch.Tensor:
        """Cast a primary neural input to ``precision_model``, promoted with the active autocast dtype.

        Autocast runs eligible operators in its dtype, so rounding the input below it would only lose precision,
        e.g. a JVP's finite-difference stencil under FP32 autocast. Autocast leaves FP64 alone, so wider inputs are
        still narrowed. ``None`` casts nothing.
        """
        precision = self.precision_model
        if precision is None:
            return x
        if torch.is_autocast_enabled(x.device.type):
            precision = torch.promote_types(precision, torch.get_autocast_dtype(x.device.type))
        return x.to(dtype=precision)

    @property
    def precision_opt(self) -> Optional[torch.dtype]:
        """Root-wide storage dtype: the recorded ``precision_opt`` if given, else ``precision_model``."""
        return self.precision_model if self._precision_opt is None else self._precision_opt

    @property
    def device(self) -> Optional[torch.device]:
        """The root-wide execution device recorded by ``prepare_module``; initially ``None``."""
        return self._device

    def _fastgen_modules(self) -> Iterator["FastGenModule"]:
        """Yield every FastGen module in the registered tree, including this one, once each."""
        return (module for module in self.modules() if isinstance(module, FastGenModule))

    def _named_fastgen_modules(self) -> Iterator[Tuple[str, "FastGenModule"]]:
        """Yield ``(prefix, module)`` for every FastGen module in the registered tree, including this one."""
        return ((prefix, module) for prefix, module in self.named_modules() if isinstance(module, FastGenModule))

    def _nearest_nested_modules(self) -> Iterator["FastGenModule"]:
        """Yield the nearest nested FastGen modules once each, without descending into them."""
        seen: Set[int] = set()
        stack = list(self.children())
        while stack:
            module = stack.pop()
            if isinstance(module, FastGenModule):
                if id(module) not in seen:
                    seen.add(id(module))
                    yield module
            else:
                stack.extend(module.children())

    def _set_device(self, device: torch.device | str | None) -> None:
        """Record the root-wide execution device; ``prepare_module`` owns placement and sharding."""
        target = None if device is None else torch.device(device)
        for module in self._fastgen_modules():
            module._device = target

    def _fixed_precision_overrides(self) -> Dict[str, torch.dtype]:
        """Return exact qualified state names whose dtype is fixed independently of the default."""
        return {}

    def _record_precision(
        self,
        precision_model: Optional[torch.dtype] = None,
        precision_opt: Optional[torch.dtype] = None,
    ) -> None:
        """Record both precision requests on every FastGen module; each resolves its own overrides."""
        for module in self._fastgen_modules():
            module._precision_model = precision_model
            module._precision_opt = precision_opt
            if module.precision_model != precision_model:
                logger.info(
                    f"{type(module).__name__} overrides precision_model from {precision_model} "
                    f"to {module.precision_model}."
                )

    def _set_precision(
        self,
        precision_model: Optional[torch.dtype] = None,
        precision_opt: Optional[torch.dtype] = None,
    ) -> None:
        """Record the precision requests, then cast the state every FastGen module owns in one pass.

        Storage follows ``precision_opt`` when given, else each module's effective ``precision_model``;
        ``_fixed_precision_overrides`` pin exact owned tensors. Shared state must resolve to one dtype
        in every owning module. Casting must precede FSDP wrapping.
        """
        self._record_precision(precision_model, precision_opt)
        entries = []
        for prefix, module in self._named_fastgen_modules():
            overrides = module._fixed_precision_overrides() if module.precision_opt is not None else {}
            owned = resolve_storage_dtypes(
                module, module.precision_opt, overrides, exclude=module._nearest_nested_modules()
            )
            entries.extend((f"{prefix}.{name}" if prefix else name, tensor, dtype) for name, tensor, dtype in owned)
        cast_to_storage_dtypes(self, entries)

    def _set_train_mode(self, trainable: bool) -> None:
        """Set training mode and gradients: this module's policy over its tree, then each nested module's own.

        ``nn.Module.requires_grad_`` sets parameter flags directly and never calls a nested module's
        override, so the nearest nested FastGen modules re-apply their own policy afterwards and win
        for the state they own, mirroring ``_set_precision``.
        """
        frozen = [(name, p) for name, p in self.named_parameters() if not p.requires_grad] if trainable else []
        self.train(trainable)
        self.requires_grad_(trainable)
        for child in self._nearest_nested_modules():
            child._set_train_mode(trainable)
        newly_trainable = [name for name, parameter in frozen if parameter.requires_grad]
        if newly_trainable:
            logger.warning(
                f"{type(self).__name__}: The `requires_grad` attribute of these parameters was `False` "
                f"and has been set to `True`: {newly_trainable}. Override `requires_grad_` to keep them frozen."
            )

    def _validate_preparation_request(
        self,
        device: torch.device | str | None,
        precision_model: Optional[torch.dtype],
        precision_opt: Optional[torch.dtype],
        trainable: bool,
        fsdp_context: Optional["FSDPContext"],
    ) -> None:
        """Validate local state and request agreement across ranks before any mutation."""
        # Even placement-free requests must participate: a peer may have requested placement.
        with raise_across_ranks(f"prepare_module of {type(self).__name__}"):
            for module in self._fastgen_modules():
                if module.is_prepared:
                    raise RuntimeError(
                        f"{type(module).__name__} is already prepared; modules can only be prepared once. "
                        "Construct a fresh module instead."
                    )
            reject_inference_state(self)
            reject_extra_state(self)
            reject_complex_parameters(self)
            reject_unsafe_placement(self, "module preparation")
            if device is None and has_meta_state(self):
                raise RuntimeError(
                    f"{type(self).__name__} holds meta tensors; preparing it without a device or an FSDP context "
                    "would leave it unmaterialized."
                )
            # Parsing a device can fail on one rank only, so build the request inside the guard.
            device_type = None if device is None else torch.device(device).type
            request = (device_type, precision_model, precision_opt, trainable, fsdp_context is not None)
        validate_object_across_ranks(request, operation="prepare_module request")

    def prepare_module(
        self,
        device: torch.device | str | None,
        precision_model: Optional[torch.dtype] = None,
        precision_opt: Optional[torch.dtype] = None,
        trainable: bool = False,
        fsdp_context: Optional["FSDPContext"] = None,
    ):
        """Record precisions and device, cast storage, set mode and gradients, then place or shard.

        A collective lifecycle step: every rank prepares each module once, in the same order, and
        all requests must agree. ``trainable`` selects train mode and gradients; storage follows
        ``precision_opt`` or else ``precision_model``. ``fsdp_context`` shards the root and owns its
        physical placement, so its execution device replaces ``device``. Otherwise ``device`` places
        the complete tree, and ``None`` leaves placement to the caller. Validation precedes mutation;
        a later failure leaves ``is_prepared=False`` without undoing casts or partial sharding.
        See the module docstring for the full lifecycle contract.
        """
        if fsdp_context is not None:
            device = fsdp_context.device
        self._validate_preparation_request(device, precision_model, precision_opt, trainable, fsdp_context)

        # Finish local casting on every rank before anyone enters placement/sharding collectives.
        with raise_across_ranks(f"prepare_module precision and mode of {type(self).__name__}"):
            self._set_precision(precision_model, precision_opt)
            self._set_train_mode(trainable)
            self._set_device(device)

        if fsdp_context is not None:
            fsdp_context._shard_module(self)
        elif device is not None:
            move_and_sync_module(self, device=device)
        # Every step above fails on all ranks together, so markers commit only once all ranks succeed.
        self._is_fsdp = fsdp_context is not None
        for module in self._fastgen_modules():
            module._is_prepared = True
        return self
