# Helper Script Coverage

The Hugging Face model resolver is covered by
`tests/unit_test/tool/agent_skill_checks/conversion_templates_test.py`.

The focused tests cover explicit local/Hub source selection, model and dataset
repository types, bare relative local paths, exit-zero cache misses, and one
commit-pinned cache-aware resolution per authorized artifact type.
