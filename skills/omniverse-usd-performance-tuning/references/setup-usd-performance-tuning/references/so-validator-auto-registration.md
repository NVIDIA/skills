# SO Validator Auto-Registration

Shared reference for how the Usd Optimize (SO) performance validator rules
register into the Omni Asset Validator (OAV). Cited by both
`install-usd-optimize-standalone/README.md` and
`install-usd-validation-nvidia-standalone/README.md`.

The standalone SO package includes `omni.scene.optimizer.validators` — 25 Python
validator rules (mesh density, unused UVs, primitive fit, etc.) declared with
`@register_rule` decorators. When OAV and the Usd Optimize package share the same
Python environment, importing the validators auto-registers them; no
`register_all()` call is needed for rule discovery — the decorators handle
registration at import time.

```python
import omni.scene.optimizer.validators  # triggers @register_rule decorators

from omni.asset_validator import CategoryRuleRegistry
registry = CategoryRuleRegistry()
# Now includes the "Usd:Performance" and "Omni:Geometry" categories
```

Expected: the `Usd:Performance` and `Omni:Geometry` categories appear with ~25
additional rules.

**Category names confirm discovery only — they are not validation scope.** Do not
select rules by bare name: `usd-validation-runner` selects validators by canonical
concept and resolves them to rule classes by identity (via
`scripts/usd_validation_executor.py`) before calling `enable_rule()`. A bare
`find_rule()` cannot tell the Usd Optimize and usd-validation-nvidia rules that
share a class name apart.
