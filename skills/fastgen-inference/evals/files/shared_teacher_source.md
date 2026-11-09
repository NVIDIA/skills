# FastGen shared teacher source evidence

Source revision: `e0a92619fc56daf4dfa081273c7729c045bf1dcc`. This exact excerpt is input evidence, not an installed checkout or an executed test.

## [fastgen/networks/lora_utils.py:272](https://github.com/NVlabs/FastGen/blob/e0a92619fc56daf4dfa081273c7729c045bf1dcc/fastgen/networks/lora_utils.py#L272)

```python
class SharedBaseTeacher(torch.nn.Module):
    """Teacher view over an adapter student: the same network run as the base model.

    Holds the net by reference under ``base_net`` (never registered), so the view
    owns no parameters and stays out of checkpoints, FSDP dicts, and cast sweeps.
    ``forward`` and every callable reached through ``__getattr__`` run as the base
    model (adapters off, eval; both restored on exit) -- anything invoked on a
    teacher is by definition a base-model computation. Other lookups pass through.
    """

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        # Bypass Module registration: the net's parameters must not become ours.
        object.__setattr__(self, "base_net", net)
        self._base_model_active = False

    @contextlib.contextmanager
    def _as_base_model(self):
        # A delegated guidance recipe calls back through this provider for each
        # branch. Keep adapters disabled until the outermost teacher call exits.
        if self._base_model_active:
            # The outer call resolved checkpointing for its own grad mode; this call may have enabled grad.
            with temporarily_disable_activation_checkpointing(self.base_net):
                yield
            return
        net = self.base_net
        self._base_model_active = True
        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(train_mode(net, mode=False))
                stack.enter_context(net.adapters_disabled())
                # Student-only attribute defaults (e.g. an embedded-guidance scale)
                # must not steer base-model computations.
                for name, value in (net.base_model_overrides or {}).items():
                    stack.enter_context(attr_override(net, name, value))
                stack.enter_context(temporarily_disable_activation_checkpointing(net))
                yield
        finally:
            self._base_model_active = False

    def forward(self, *args, **kwargs):
        with self._as_base_model():
            return self.base_net(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        attr = getattr(self.base_net, name)
        # Sub-modules (callable too) pass through raw -- wrapping would hide their
        # attributes; only methods/functions run under the base-model context.
        if callable(attr) and not isinstance(attr, torch.nn.Module):

            @functools.wraps(attr)
            def as_base_model(*args, **kwargs):
                with self._as_base_model():
                    return attr(*args, **kwargs)

            return as_base_model
        return attr
```
