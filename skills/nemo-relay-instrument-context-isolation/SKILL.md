---
name: nemo-relay-instrument-context-isolation
description: Use this skill when concurrent requests, async tasks, threads, workers, goroutines, or agents need independent NeMo Relay scope stacks and correct ancestry propagation.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation and Affiliates
---

# Use Context Isolation

Use this skill when an application runs concurrent requests, worker pools, async
tasks, goroutines, or multiple agents in the same process.
Treat scope-stack ownership as part of the request boundary.

## Core Rule

Each independent request, agent, or workflow needs its own scope stack. Do not
share one mutable stack across unrelated concurrent work unless you want shared
ancestry and shared scope-local middleware.

## Embedded Scope Model

- A root scope is always present, and pushed scopes form a parent-child tree
  beneath it.
- Scope hierarchy determines event parentage, lifetime boundaries, and
  visibility for scope-local middleware and subscribers.
- Standard scope types include `Agent`, `Function`, `Tool`, `Llm`,
  `Retriever`, `Embedder`, `Reranker`, `Guardrail`, `Evaluator`, `Custom`, and
  `Unknown`.
- Scope start and end events can carry semantic `input` and `output` payloads
  when the scope represents a request, task, or meaningful result boundary.
- Scope-local registrations disappear when the owning scope closes; use them
  for behavior that should not outlive a request or agent run.
- Mark events are useful for retries, checkpoints, interrupts, and important
  state transitions that are not full spans.

## Per-Language Defaults

- **Python**: rely on task-local behavior via `get_scope_stack()` and
  `contextvars`, or explicitly propagate when work leaves the current execution
  context
- **Rust core**: use runtime helpers such as `create_scope_stack()`,
  `current_scope_stack()`, and `set_thread_scope_stack(...)` when an integration
  needs explicit stack ownership
- **Go**: use `NewScopeStack()` and `ScopeStack.Run(...)` for goroutine-safe
  usage
- **Node.js**: create and set a scope stack explicitly for the current execution
  path with `createScopeStack()` and `setThreadScopeStack(...)`

## Common Failures

- Events from different requests appear under one root UUID
- Scope-local middleware leaks across requests
- Worker-thread work runs without the expected active scope
- Integrations activate NeMo Relay without an explicitly initialized stack
- Relying on a thread-local stack after crossing async tasks, goroutines, or JS
  worker boundaries

## Related Skills

- `nemo-relay-instrument-calls`
- `nemo-relay-plugin-observability`
- `nemo-relay-debug-runtime-integration`
