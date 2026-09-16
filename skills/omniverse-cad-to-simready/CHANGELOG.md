# CAD-to-SimReady Changelog

## 0.2.0 (Unreleased)

This release improves the reliability, portability, dependency management, and
security-scanner compatibility of the `omniverse-cad-to-simready` workflow.

### Added

- Configurable Content Agents deployment for NVIDIA, OpenAI-compatible,
  Anthropic, and Google/Gemini model providers.
- Deterministic upstream dependency pins and a checked-in tested-version lock
  file for conversion, validation, rendering, and Content Agents integrations.
- Expanded preflight reporting, runtime validation, and evaluation coverage.
- An audited SkillSpector baseline using exact fingerprints for reviewed false
  positives. New or changed findings continue to fail closed.

### Changed

- Migrated CAD conversion to the self-contained `usd-convert-cad` 0.2.0 wheel
  in an isolated Python 3.12 environment.
- Simplified the top-level workflow router while preserving detailed
  stage-specific guidance in nested references.
- Improved conversion-only routing so Content Agents deployment is skipped when
  material and physics assignment are not requested.
- Updated the CAD-to-SimReady skill and skill-card version to 0.2.0.

### Fixed

- Hardened credential environment-variable validation, forwarding, redaction,
  and generated configuration references.
- Improved reuse and validation of existing Content Agents endpoints, including
  local Docker host translation.
- Rejected missing, blank, or uniform render and thumbnail outputs instead of
  accepting them as successful inspections.
- Correctly counted geometry represented through USD instance proxies.
- Improved SimReady validation, optimizer recovery, dependency checks, and
  actionable error reporting.

### Migration Notes

- Use Python 3.12 for the workflow and its managed environments.
- Install and invoke `usd-convert-cad` 0.2.0 as a wheel. The previous KAT/Kit
  fallback is no longer supported.
- Treat `upstream-versions.lock.json` as the tested dependency source of truth;
  preflight blocks on incompatible direct or transitive package pins.
