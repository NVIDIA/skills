# SO Validator Registration

Shared reference for how the Usd Optimize (SO) performance validator rules get
into the Omni Asset Validator (OAV) rule registry. Cited by both
`install-usd-optimize-standalone/README.md` and
`install-usd-validation-nvidia-standalone/README.md`.

**Importing the package does not register the rules. Call `register_all()`.**
An earlier version of this file claimed the opposite — that the `@register_rule`
decorators fire at import time — and that claim was the origin of a
release-blocking bug: `usd_validation_executor.py` did a bare
`import omni.scene.optimizer.validators` and every SO validator concept then
failed to resolve, because the registry held only the OAV rules.

## What actually happens on usd-optimize 1.0.4

`usd_optimize/validators/__init__.py` is plain re-export imports plus a
module-level `_RULE_CATEGORIES` tuple of 25 `(rule, category)` pairs. The
`@register_rule` decorator is applied inside `register_all()`, not at module
scope:

```python
def register_all():
    """Register Usd Optimize rules with Asset Validator."""
    registry = CategoryRuleRegistry()
    for rule, category in _RULE_CATEGORIES:
        if registry.get_category(rule) is None:
            register_rule(category)(rule)
    return [rule for rule, _ in _RULE_CATEGORIES]
```

Measured on usd-validation-nvidia 1.20.0 + usd-optimize 1.0.4: the registry holds
**40** rules after importing the SO package and **65** after `register_all()` —
**25** SO rules, matching `len(_RULE_CATEGORIES)`. The `if ... is None` guard makes
`register_all()` idempotent, so calling it on every enumeration is safe.

```python
import usd_optimize.validators as V   # import alone registers nothing
V.register_all()                      # this is what registers the 25 rules

from omni.asset_validator import CategoryRuleRegistry
registry = CategoryRuleRegistry()
# Now includes the "Usd:Performance" and "Omni:Geometry" categories
```

`omni.scene.optimizer.validators` is a deprecated alias of the same module object
(`omni.scene.optimizer.validators is usd_optimize.validators` is `True`), so
either name works; prefer `usd_optimize.validators` in new code. The rule classes
report `__module__ = usd_optimize.validators.*` under both names, because
`__module__` is the defining module and no import alias changes it.

## Why entry points do not cover this deployment shape

usd-validation-nvidia does ship a plugin system: `usd_validation_nvidia/_plugins.py`
defines a `PluginManager` that discovers registrants through the
`usd_validation_nvidia` entry-point group and calls `on_startup()` /
`on_shutdown()` on each. That is the mechanism a pip-installed provider would
use to self-register, and if Usd Optimize were pip-installed with such an entry
point declared, no explicit call would be needed.

It cannot apply here. Entry-point discovery goes through `importlib.metadata`,
which reads **installed distribution metadata**. The skill consumes Usd Optimize
as an extracted release zip placed on `PYTHONPATH`, and that tree contains no
`.dist-info`, no `.egg-info`, and no `entry_points.txt` anywhere —
`importlib.metadata.version("usd-optimize")` raises `PackageNotFoundError`. The
package is structurally invisible to entry-point discovery regardless of what
upstream declares in its own build config. Confirmed on this runtime: the
`usd_validation_nvidia` entry-point group resolves to an empty list.

So explicit registration is not a workaround for an upstream omission. In the
extracted-zip deployment it is the only mechanism available.

| Deployment shape | How SO rules register |
|---|---|
| Extracted release zip on `PYTHONPATH` (what this skill uses) | Explicit `register_all()`. No distribution metadata exists, so entry points cannot fire. |
| Pip-installed distribution that declares a `usd_validation_nvidia` entry point | The plugin manager can register it at OAV startup. `register_all()` remains safe and idempotent. |
| Kit extension | Kit's own extension startup registers the rules; `register_all()` remains safe and idempotent. |

Calling `register_all()` is correct in all three, which is why
`usd_validation_executor.py` calls it unconditionally (tolerantly, trying
`usd_optimize.validators` then `omni.scene.optimizer.validators`) on every rule
enumeration rather than branching on deployment shape.

## Selection is still by identity, never by name

**Category names confirm discovery only — they are not validation scope.** Do not
select rules by bare name: `usd-validation-runner` selects validators by canonical
concept and resolves them to rule classes by identity (via
`scripts/usd_validation_executor.py`) before calling `enable_rule()`. A bare
`find_rule()` cannot tell the Usd Optimize and usd-validation-nvidia rules that
share a class name apart — `IndexedPrimvarChecker` is registered by both, and the
two differ by roughly three orders of magnitude in runtime.

Identity is resolved as `(provider family of module, class_name)`, so a module
rename inside a family still resolves while a cross-family name collision never
does. See `usd-validation-runner/references/validator-concepts.json`.
