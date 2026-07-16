---
name: nemo-relay-instrument-typed-wrappers
description: Use this skill when adding NeMo Relay typed wrappers, domain types, or provider codecs while preserving JSON middleware semantics and caller-visible behavior.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation and Affiliates
---

# Use Typed Wrappers And Codecs

Use this skill when an application wants stronger domain types than raw JSON for
tool or LLM integration.
Keep typed boundaries explicit so middleware still sees predictable JSON.

## Default Guidance

- Prefer plain JSON first for initial adoption.
- Reach for typed wrappers when the application already has stable domain models.
- Keep in mind that middleware still operates on JSON, not typed objects.

## Embedded Codec Model

- A typed value codec is a pure boundary translator. It converts
  application-facing values to JSON before NeMo Relay emits events or runs
  middleware, then converts JSON back into the framework callback or caller type.
- Python exposes `JsonPassthrough`, `DataclassCodec`, `PydanticCodec`, and
  `BestEffortAnyCodec`. Node.js exposes `JsonPassthrough` plus custom
  `Codec<T>` implementations.
- Use `BestEffortAnyCodec` only at boundaries where strict schemas are not
  available. Prefer dataclass, Pydantic, or explicit Node.js codecs when the
  framework owns a stable schema.
- Provider codecs are different from typed value codecs: they normalize
  provider-specific LLM requests and responses so middleware and subscribers can
  inspect messages, tools, model names, generation parameters, and response
  annotations.
- Built-in provider codecs include `OpenAIChatCodec`, `OpenAIResponsesCodec`,
  and `AnthropicMessagesCodec` in Python, Node.js, and Rust. Choose the
  codec that matches the actual provider payload shape.
- Response codecs annotate LLM end events with fields such as `id`, `model`,
  `message`, `tool_calls`, `finish_reason`, `usage`, provider-specific data, and
  extra unmodeled fields. They do not rewrite the caller-visible response.
- Request codecs run before LLM request intercepts. Intercepts receive both the
  raw `LLMRequest` and optional annotated request; `encode` merges annotated
  edits back before execution intercepts and the provider callback run.

## Key Rules

- Typed wrappers are currently a first-class path for Python and Node.js; Rust
  uses codec traits directly
- Request/response conversion belongs in codecs
- Intercepts and guardrails see JSON values after encoding
- Changes made by middleware survive into the decode step

## Choose A Codec

- `JsonPassthrough` for JSON-native values
- `DataclassCodec` or `PydanticCodec` in Python when the models already exist
- Custom codecs for domain-specific wire shapes
- `BestEffortAnyCodec` only when broad flexibility is worth the looser contract
- Provider codecs for LLM provider payloads, not application domain objects
  conversion

## Validation Checklist

- [ ] Codec output is JSON-compatible
- [ ] Required fields survive `toJson`/`fromJson` or `decode`/`encode`
- [ ] Middleware sees the expected serialized shape
- [ ] Provider codecs preserve fields they do not understand
- [ ] Response codec failures do not break the underlying LLM call
- [ ] Request codec `encode` preserves original provider fields unless an
  intercept intentionally changes them

## Related Skills

- `nemo-relay-instrument-calls`
- `nemo-relay-plugin-observability`
- `nemo-relay-debug-runtime-integration`
