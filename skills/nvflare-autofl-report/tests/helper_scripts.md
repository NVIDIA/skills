# Helper Script Checks

The report helper is covered by
`tests/unit_test/tool/autofl_skill_report_test.py`.

Run the focused checks from the repository root:

```bash
pytest tests/unit_test/tool/autofl_skill_report_test.py -q
```

The tests exercise stopped-state admission, pending-candidate refusal, explicit
interruption confirmation, retained versus observed result selection,
Markdown/JSON artifacts with optional plot availability, min/max objectives,
literature-event linkage, candidate lineage, malformed contract sections,
per-run metric provenance, budget and test-metric warnings, guard/plotter
parity, optional agent context, and operation outside a Git repository. The
suite rejects legacy minimization evidence without provenance and checks
campaign-state/ledger accounting, portable path aliases,
read-only evidence locking, and the real merged Auto-FL plotter interface.
