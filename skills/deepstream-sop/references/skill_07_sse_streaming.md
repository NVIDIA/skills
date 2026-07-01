# § 7 — SSE Streaming Pattern

> **Generates**: Part of `nvds_action_detector/api_server.py`
> **Critical Rules**: CLEANUP_ON_DISCONNECT
> **Pipeline stage**: Output — converts SOPVideoProcessor results into SSE events or
> non-streaming JSON responses

---

## Architecture Context

```
SOPVideoProcessor (§ 6)
    final_queue ──► ds_sop_generate_chunks() ──► _run_stream_chunks_response()
                         │                              │
                    async generator               SSE formatter
                    polls final_queue             yields "data: {...}\n\n"
                    detects disconnect                  │
                         │                              ▼
                    try/finally                  StreamingResponse
                    stops processor              (text/event-stream)
                    (releases camera)                   │
                                                       ▼
                                                   Client (curl -N)
```

---

## SSE Generator with Cleanup

```python
async def ds_sop_generate_chunks(request, raw_request, processor) -> AsyncGenerator[dict, None]:
    """Async generator yielding chunk dicts from the SOPVideoProcessor final_queue.

    CRITICAL [CLEANUP_ON_DISCONNECT]: Always stops the processor on exit (disconnect, completion,
    or error) to release hardware resources like Basler cameras. Without this cleanup,
    the camera pipeline keeps running and holds the GigE Vision hardware lock.

    Flow:
      1. Poll raw_request.receive() to detect client disconnect (non-blocking)
      2. Poll processor.final_queue for chunk results (0.3s timeout)
      3. Yield each chunk to _run_stream_chunks_response() for SSE formatting
      4. On exit (any reason): stop the processor via trigger_stop_processors()
    """
    loop = asyncio.get_event_loop()
    try:
        while True:
            # Check for client disconnect (non-blocking with timeout)
            try:
                disconnect = await asyncio.wait_for(raw_request.receive(), timeout=0.01)
                if disconnect["type"] == "http.disconnect":
                    break
            except asyncio.TimeoutError:
                pass

            # Poll final_queue with 0.3s timeout
            try:
                chunk = await loop.run_in_executor(
                    None, lambda: processor.final_queue.get(block=True, timeout=0.3)
                )
            except queue.Empty:
                continue

            yield chunk
            if chunk is None or chunk.get("is_last_item", False):
                break
    finally:
        # Always stop the processor to release the camera/pipeline
        global_sop_manager.trigger_stop_processors(processor, force=True)
```

---

## SSE Response Formatter

```python
async def _run_stream_chunks_response(chunks_generator, request, raw_request):
    """Convert chunk dicts from the 4-stage pipeline into SSE chat.completion.chunk format.

    Each chunk from SOPVideoProcessor contains:
      - response: VLM classification text (Stage 3)
      - start_time, end_time: chunk temporal boundaries (Stage 2)
      - sop_check: SOP validation result (Stage 4, if enabled)
      - req_id: unique VLM request ID
    """
    chat_id = None
    async for chunk in chunks_generator:
        if chunk is None:
            break
        delta = DeltaMessage(content=chunk.get("response", "").strip())
        choice = ChatCompletionResponseStreamChoice(index=0, delta=delta, chunk_metadata=chunk)
        if chat_id is None:
            chat_id = f"chatcmpl-{chunk.get('req_id', uuid.uuid4().hex)}"
        response = ChatCompletionStreamResponse(
            id=chat_id, created=int(time.time()), model=request.model, choices=[choice]
        )
        yield f"data: {response.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"

# In the endpoint:
if request.stream:
    stream_gen = _run_stream_chunks_response(chunks_gen, request, raw_request)
    return StreamingResponse(content=stream_gen, media_type="text/event-stream")
```

> Non-streaming (`stream=false`) collects chunks into one `ChatCompletionResponse`; default `chat_id = f"chatcmpl-{uuid.uuid4().hex}"` when no chunk sets it (client disconnect / empty input) so `id` is never `None`.

---

## API_DUMMY_TEST Streaming (CRITICAL — test suite will fail if missed)

When `API_DUMMY_TEST=true`, the chat endpoint **must still return SSE format** for
`stream=true`. The test suite asserts `"text/event-stream" in content-type` and
parses `data:` lines — returning plain JSON for streaming in dummy mode fails
`test_chat_completion_streaming`.

```python
if API_DUMMY_TEST:
    if request.stream:
        # MUST return SSE format even in dummy mode — test suite expects it
        return StreamingResponse(
            content=_dummy_stream_response(request),
            media_type="text/event-stream",
        )
    return _dummy_response(request)

async def _dummy_stream_response(request) -> AsyncGenerator[str, None]:
    """Generate a single dummy SSE chunk for API_DUMMY_TEST mode."""
    chat_id = f"chatcmpl-{uuid.uuid4().hex}"
    delta = DeltaMessage(content="Dummy response -- API_DUMMY_TEST=true")
    choice = ChatCompletionResponseStreamChoice(index=0, delta=delta, finish_reason="stop")
    chunk = ChatCompletionStreamResponse(
        id=chat_id, created=int(time.time()),
        model=request.model or "ds_sop_model", choices=[choice],
    )
    yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
```
