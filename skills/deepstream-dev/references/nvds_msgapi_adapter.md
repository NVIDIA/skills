# DeepStream Messaging API Adapter

## Table of Contents

1. [nvds_msgapi Interface](#nvds_msgapi-protocol-adapter-interface)
2. [Implementation Patterns & Skeleton](#building-a-custom-protocol-adapter)
3. [GStreamer Integration](#integrating-with-gst-nvmsgbroker)
4. [Configuration File Format](#configuration-file-format)
5. [nvds_logger](#nvds_logger)
6. [Common Pitfalls](#common-pitfalls-and-troubleshooting)
7. [Checklist](#checklist-for-custom-adapter-development)

---

## Overview

Build a custom messaging adapter for any protocol (NATS, ZeroMQ, Pulsar, custom TCP) by implementing `nvds_msgapi.h` and compiling as a shared library (`.so`).

---

## nvds_msgapi: Protocol Adapter Interface

### Types and Callbacks

Key types/enums from `nvds_msgapi.h` (excerpt):

```c
/** Defines the handle used by messaging API functions. */
typedef void *NvDsMsgApiHandle;
/**
 * Defines events associated with connections to remote entities.
 */
typedef enum {
  /** Specifies that a connection attempt was successful. */
  NVDS_MSGAPI_EVT_SUCCESS,
  /** Specifies disconnection of a connection handle. */
  NVDS_MSGAPI_EVT_DISCONNECT,
  /** Specifies that the remote service is down. */
  NVDS_MSGAPI_EVT_SERVICE_DOWN
} NvDsMsgApiEventType;

/**
 * Defines completion codes for operations in the messaging API.
 */
typedef enum {
  NVDS_MSGAPI_OK,
  NVDS_MSGAPI_ERR,
  NVDS_MSGAPI_UNKNOWN_TOPIC
} NvDsMsgApiErrorType;

typedef void (*nvds_msgapi_connect_cb_t)(NvDsMsgApiHandle h_ptr, NvDsMsgApiEventType ds_evt);

typedef void (*nvds_msgapi_send_cb_t)(void *user_ptr, NvDsMsgApiErrorType completion_flag);

typedef void (*nvds_msgapi_subscribe_request_cb_t)(NvDsMsgApiErrorType flag,
                                                    void *msg,
                                                    int msg_len,
                                                    char *topic,
                                                    void *user_ptr);
```

### Functions to Export

---

#### nvds_msgapi_connect() — Create a Connection

```c
NvDsMsgApiHandle nvds_msgapi_connect(char *connection_str,
                                     nvds_msgapi_connect_cb_t connect_cb,
                                     char *config_path);
```

| Parameter | Type | Description |
|---|---|---|
| `connection_str` | `char *` | Connection parameters in adapter-defined format. NVIDIA convention is `"url;port;topic"`, but the adapter may use any format or accept `NULL`. |
| `connect_cb` | `nvds_msgapi_connect_cb_t` | Callback for events associated with the connection. Invoke with `NVDS_MSGAPI_EVT_SUCCESS` on successful connect, `NVDS_MSGAPI_EVT_DISCONNECT` or `NVDS_MSGAPI_EVT_SERVICE_DOWN` on disconnect/error. |
| `config_path` | `char *` | Path to a configuration file passed to the protocol adapter. May be `NULL`. |

**Return**: `NvDsMsgApiHandle` on success, `NULL` on failure.

---

#### nvds_msgapi_send() — Synchronous Send

```c
NvDsMsgApiErrorType nvds_msgapi_send(NvDsMsgApiHandle h_ptr,
                                     char *topic,
                                     const uint8_t *payload,
                                     size_t nbuf);
```

| Parameter | Type | Description |
|---|---|---|
| `h_ptr` | `NvDsMsgApiHandle` | Connection handle. |
| `topic` | `char *` | Topic to send the message to. May be `NULL`. |
| `payload` | `const uint8_t *` | Pointer to the message bytes. The message may but need not be a NULL-terminated string. |
| `nbuf` | `size_t` | Number of bytes to send, including the terminating NULL if the message is a string. |

**Return**: Completion code for the send operation.

---

#### nvds_msgapi_send_async() — Asynchronous Send

```c
NvDsMsgApiErrorType nvds_msgapi_send_async(NvDsMsgApiHandle h_ptr,
                                           char *topic,
                                           const uint8_t *payload,
                                           size_t nbuf,
                                           nvds_msgapi_send_cb_t send_callback,
                                           void *user_ptr);
```

| Parameter | Type | Description |
|---|---|---|
| `h_ptr` | `NvDsMsgApiHandle` | Connection handle. |
| `topic` | `char *` | Topic name. May be `NULL`. |
| `payload` | `const uint8_t *` | Pointer to the message bytes. The message may but need not be a NULL-terminated string. |
| `nbuf` | `size_t` | Number of bytes to send, including the terminating NULL if the message is a string. |
| `send_callback` | `nvds_msgapi_send_cb_t` | Callback invoked when the operation completes: `void (*)(void *user_ptr, NvDsMsgApiErrorType completion_flag)`. |
| `user_ptr` | `void *` | Context pointer forwarded verbatim to `send_callback`. |

**Return**: Completion code for the send operation.

---

#### nvds_msgapi_do_work() — Periodic Work

```c
void nvds_msgapi_do_work(NvDsMsgApiHandle h_ptr);
```

Allows the adapter to execute underlying protocol logic — service pending incoming and outgoing messages, perform periodic housekeeping tasks such as sending heartbeats. The client must call this periodically, according to the adapter's requirements. If the adapter uses its own I/O threads, this can be a no-op.

---

#### nvds_msgapi_disconnect() — Terminate a Connection

```c
NvDsMsgApiErrorType nvds_msgapi_disconnect(NvDsMsgApiHandle h_ptr);
```

Terminates a connection. The adapter must release all resources associated with `h_ptr` and must not use the handle again after this call returns.

**Return**: Completion code for the terminate operation.

---

#### nvds_msgapi_getversion() / nvds_msgapi_get_protocol_name()

```c
char *nvds_msgapi_getversion(void);
char *nvds_msgapi_get_protocol_name(void);
```

| Function | Description |
|---|---|
| `nvds_msgapi_getversion()` | Returns the messaging API version string supported by the adapter (e.g., `"2.0"` in `"major.minor"` format). |
| `nvds_msgapi_get_protocol_name()` | Returns the name of the protocol used in the adapter (e.g., `"KAFKA"`, `"MQTT"`). |

---

#### nvds_msgapi_subscribe() — Subscribe

```c
NvDsMsgApiErrorType nvds_msgapi_subscribe(NvDsMsgApiHandle h_ptr,
                                          char **topics,
                                          int num_topics,
                                          nvds_msgapi_subscribe_request_cb_t cb,
                                          void *user_ctx);
```

Subscribes to a remote entity for receiving messages on particular topic(s). The adapter must invoke `cb(flag, msg, msg_len, topic, user_ctx)` for each incoming message on subscribed topics.

The subscribes API **MUST** be implemented; it may be used in the `libnvds_msgbroker.so`.
Multiple topic subscriptions **MUST** be supported.

| Parameter | Type | Description |
|---|---|---|
| `h_ptr` | `NvDsMsgApiHandle` | Connection handle. |
| `topics` | `char **` | Array of topic strings to subscribe for messages. |
| `num_topics` | `int` | Number of topics in the `topics` array. |
| `cb` | `nvds_msgapi_subscribe_request_cb_t` | Callback invoked for each incoming message: `void (*)(NvDsMsgApiErrorType flag, void *msg, int msg_len, char *topic, void *user_ptr)`. Reports consumption status and the received message/payload. |
| `user_ctx` | `void *` | Opaque pointer forwarded verbatim to `cb`. |

**Return**: `NVDS_MSGAPI_OK` on success.

---

#### nvds_msgapi_connection_signature() — Connection Sharing

```c
NvDsMsgApiErrorType nvds_msgapi_connection_signature(char *broker_str,
                                                     char *cfg,
                                                     char *output_str,
                                                     int max_len);
```

Generates a unique connection signature by parsing `broker_str` and `cfg`. A connection signature is a unique string used to identify a connection. It can be retrieved only if the `share-connection` config option is set to `1`.

| Parameter | Type | Description |
|---|---|---|
| `broker_str` | `char *` | Broker connection string used to create the connection. |
| `cfg` | `char *` | Path to the adapter config file. |
| `output_str` | `char *` | **Output**: buffer to write the connection signature string into. |
| `max_len` | `int` | Maximum length of `output_str` buffer. |

**Return**: Valid connection signature in `output_str` on success. Empty string (`""`) in case of errors or if `share-connection` config option is not set to `1`. The signature should be a deterministic function of `broker_str` and `cfg` (e.g., a hash or concatenation of key fields).

---

## Building a Custom Protocol Adapter

### Choosing Your Implementation Pattern

| Pattern | Library characteristics | Built-in reference |
|---------|------------------------|-------------------|
| **A — Library owns threads** | SDK fires per-message callback from its own thread. `do_work()` is a no-op. | Azure IoT SDK |
| **B — Event loop + opaque pointer** | Library has a `poll()` call; per-message opaque `void *` returned verbatim in delivery callback. | Kafka (librdkafka) |
| **C — Event loop + message-ID map** | Library returns integer message-ID from publish; delivery callback receives same ID. | MQTT (libmosquitto) |
| **D — Blocking library** | Only synchronous blocking send. No async or event-loop API. | AMQP (rabbitmq-c), Redis (hiredis) |

`connect_cb` calling convention:
- **Pattern A/C**: call with `NVDS_MSGAPI_EVT_SUCCESS` from library's own connection callback.
- **Pattern B/D**: do NOT call in `nvds_msgapi_connect()`; call with `NVDS_MSGAPI_EVT_SERVICE_DOWN` from `do_work()` or consumer thread on runtime error.

---

### Skeleton Implementation (C)

> **CRITICAL**: `send_async()` completion callback **must fire from a different thread** than the caller. Both legacy and new-api paths hold an internal mutex across `send_async()`. Calling the callback synchronously — including in the error path — causes immediate deadlock. Always return `NVDS_MSGAPI_ERR` directly without invoking the callback on error.

Skeleton below implements **Pattern B**. Adaptation notes for A/C/D follow.

```c
#include <stdlib.h>
#include <string.h>
#include "nvds_msgapi.h"
#include "nvds_logger.h"

#define LOG_CAT "DSLOG:CUSTOM_PROTO"

typedef struct {
    nvds_msgapi_send_cb_t  cb;
    void                  *user_ptr;
} PendingSend;

typedef struct {
    char *server_url;
    int   port;
    int   connected;
    /* your_client_t *client; */
    nvds_msgapi_connect_cb_t connect_cb;
    /* pthread_t consumer_tid; int stop_consumer; */
} CustomAdapterCtx;

/* Fired from your_proto_poll() inside do_work() — satisfies "different thread" requirement */
static void on_send_complete(int success, void *msg_opaque)
{
    PendingSend *ps = (PendingSend *)msg_opaque;
    if (ps->cb) ps->cb(ps->user_ptr, success ? NVDS_MSGAPI_OK : NVDS_MSGAPI_ERR);
    free(ps);
}

static void on_disconnect(void *user_data)
{
    CustomAdapterCtx *ctx = (CustomAdapterCtx *)user_data;
    if (ctx->connect_cb)
        ctx->connect_cb((NvDsMsgApiHandle)ctx, NVDS_MSGAPI_EVT_SERVICE_DOWN);
}

NvDsMsgApiHandle nvds_msgapi_connect(char *connection_str,
                                     nvds_msgapi_connect_cb_t connect_cb,
                                     char *config_path)
{
    nvds_log_open();
    CustomAdapterCtx *ctx = calloc(1, sizeof(CustomAdapterCtx));
    if (!ctx) return NULL;

    ctx->connect_cb = connect_cb;

    if (connection_str) {
        char *tmp = strdup(connection_str);
        char *host = strtok(tmp, ";"), *port = strtok(NULL, ";");
        if (host) ctx->server_url = strdup(host);
        if (port) ctx->port = atoi(port);
        free(tmp);
    }

    /* Parse config_path with GKeyFile if needed */

    /* your_proto_connect(ctx->server_url, ctx->port, on_send_complete, on_disconnect, ctx);
     * On failure: free ctx and return NULL. */

    ctx->connected = 1;
    nvds_log(LOG_CAT, LOG_INFO, "Connected to %s:%d", ctx->server_url, ctx->port);
    return (NvDsMsgApiHandle)ctx;
}

NvDsMsgApiErrorType nvds_msgapi_send(NvDsMsgApiHandle h_ptr,
                                     char *topic, const uint8_t *payload, size_t nbuf)
{
    CustomAdapterCtx *ctx = (CustomAdapterCtx *)h_ptr;
    if (!ctx || !ctx->connected || !topic || !payload || nbuf <= 0) return NVDS_MSGAPI_ERR;

    /* your_proto_send_sync(ctx->client, topic, payload, nbuf); */

    return NVDS_MSGAPI_OK;
}

NvDsMsgApiErrorType nvds_msgapi_send_async(NvDsMsgApiHandle h_ptr,
                                           char *topic, const uint8_t *payload, size_t nbuf,
                                           nvds_msgapi_send_cb_t send_callback, void *user_ptr)
{
    /* Do NOT call send_callback here — deadlock. See CRITICAL note above. */
    CustomAdapterCtx *ctx = (CustomAdapterCtx *)h_ptr;
    if (!ctx || !ctx->connected || !topic || !payload || nbuf <= 0) return NVDS_MSGAPI_ERR;

    PendingSend *ps = malloc(sizeof(PendingSend));
    if (!ps) return NVDS_MSGAPI_ERR;
    ps->cb = send_callback;
    ps->user_ptr = user_ptr;

    /* your_proto_send_async(ctx->client, topic, payload, nbuf, ps);
     * Pass ps as opaque pointer — freed in on_send_complete().
     * If library doesn't copy payload internally, malloc+memcpy here and free in on_send_complete. */

    int rc = 0; /* replace with actual call */
    if (rc != 0) { free(ps); return NVDS_MSGAPI_ERR; }
    return NVDS_MSGAPI_OK;
}

void nvds_msgapi_do_work(NvDsMsgApiHandle h_ptr)
{
    CustomAdapterCtx *ctx = (CustomAdapterCtx *)h_ptr;
    if (!ctx || !ctx->connected) return;
    /* your_proto_poll(ctx->client, 0); */
}

NvDsMsgApiErrorType nvds_msgapi_subscribe(NvDsMsgApiHandle h_ptr,
                                          char **topics, int num_topics,
                                          nvds_msgapi_subscribe_request_cb_t cb, void *user_ctx)
{
    CustomAdapterCtx *ctx = (CustomAdapterCtx *)h_ptr;
    if (!ctx || !ctx->connected) return NVDS_MSGAPI_ERR;

    /* Start dedicated consumer thread (NOT the do_work thread):
     *   pthread_create(&ctx->consumer_tid, NULL, consumer_thread_fn, ctx);
     * consumer_thread_fn: blocking recv loop → cb(flag, msg, len, topic, user_ctx)
     *                     → check ctx->stop_consumer to exit */

    return NVDS_MSGAPI_OK;
}

NvDsMsgApiErrorType nvds_msgapi_disconnect(NvDsMsgApiHandle h_ptr)
{
    CustomAdapterCtx *ctx = (CustomAdapterCtx *)h_ptr;
    if (!ctx) return NVDS_MSGAPI_ERR;
    ctx->connected = 0;

    /* ctx->stop_consumer = 1; pthread_join(ctx->consumer_tid, NULL); */
    /* your_proto_flush(ctx->client); your_proto_disconnect(ctx->client); */

    free(ctx->server_url);
    free(ctx);
    nvds_log_close();
    return NVDS_MSGAPI_OK;
}

char *nvds_msgapi_getversion(void)        { return (char *)"2.0"; }
char *nvds_msgapi_get_protocol_name(void) { return (char *)"CUSTOM_PROTOCOL"; }

NvDsMsgApiErrorType nvds_msgapi_connection_signature(char *broker_str, char *cfg,
                                                     char *output_str, int max_len)
{
    if (!output_str || max_len <= 0) return NVDS_MSGAPI_ERR;
    output_str[0] = '\0';
    if (!broker_str) return NVDS_MSGAPI_ERR;

    /* Check share-connection in cfg; return OK (empty string) to disable sharing.
     *
     * Requirements:
     * - nv_msgbroker wrapper uses signature as map key for connection sharing,
     *   so use SHA-256 (fixed 64-char hex, output buffer ≥ 65 bytes)
     * - Hash PARSED connection values (host, port, credentials), not raw
     *   broker_str or config file path — identical settings from different
     *   config paths must produce the same signature
     */
    snprintf(output_str, max_len, "%s", broker_str); /* TODO: parse → SHA-256 */
    return NVDS_MSGAPI_OK;
}
```

---

### Adaptation Notes for Other Patterns

**Pattern A — Library owns threads (Azure IoT SDK)**
- `do_work()` is a no-op.
- Call `connect_cb(..., NVDS_MSGAPI_EVT_SUCCESS)` from the SDK's connection-status callback.
- In `disconnect()`, wait for in-flight sends to complete before closing the SDK handle.

```c
void nvds_msgapi_do_work(NvDsMsgApiHandle h_ptr) { /* no-op */ }
```

---

**Pattern C — Event loop + message-ID map (MQTT/libmosquitto)**
- Replace `PendingSend *` with `map[mid] = {cb, user_ptr}` protected by a mutex.
- Call `mosquitto_threaded_set(mosq, true)` before starting.
- Return your protocol name (e.g., `"MQTT"`) from `get_protocol_name()`.

```c
void nvds_msgapi_do_work(NvDsMsgApiHandle h_ptr) {
    CustomAdapterCtx *ctx = (CustomAdapterCtx *)h_ptr;
    if (!ctx || !ctx->connected) return;
    mosquitto_loop(ctx->mosq, 0, 1);
}
```

---

**Pattern D — Blocking library (AMQP/rabbitmq-c, Redis/hiredis)**
- Replace `PendingSend` with a mutex-protected queue. `send_async()` enqueues; `do_work()` atomically steals the queue and calls blocking send inline.
- Redis is the recommended Pattern D reference (uses mutex + double-buffered queue swap). AMQP uses a simplified scheme (no lock on its list) that has data-race risk under high concurrency.

```c
void nvds_msgapi_do_work(NvDsMsgApiHandle h_ptr) {
    CustomAdapterCtx *ctx = (CustomAdapterCtx *)h_ptr;
    if (!ctx || !ctx->connected) return;

    pthread_mutex_lock(&ctx->primary_mu);
    QueueNode *list = ctx->primary_head;
    ctx->primary_head = ctx->primary_tail = NULL;
    pthread_mutex_unlock(&ctx->primary_mu);

    while (list) {
        QueueNode *node = list; list = list->next;
        int rc = your_proto_send_sync(ctx->client, node->topic, node->payload, node->payload_len);
        if (node->cb) node->cb(node->user_ptr, rc == 0 ? NVDS_MSGAPI_OK : NVDS_MSGAPI_ERR);
        free(node->topic); free(node->payload); free(node);
    }
}
```

---

### Compilation

```makefile
DS_ROOT := /opt/nvidia/deepstream/deepstream

CC      := gcc
CFLAGS  := -fPIC -Wall -I$(DS_ROOT)/sources/includes $(shell pkg-config --cflags glib-2.0)
LDFLAGS := -shared $(shell pkg-config --libs glib-2.0) -lssl -lcrypto
# LDFLAGS += -lmyproto

TARGET := libnvds_custom_proto.so
SRCS   := custom_proto_adapter.c

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

install:
	cp $(TARGET) $(DS_ROOT)/lib/

clean:
	rm -f $(TARGET)

.PHONY: install clean
```

---

## Integrating with Gst-nvmsgbroker

### GStreamer Pipeline

Set the `proto-lib` property to the path of your custom adapter library:

```python
from pyservicemaker import Pipeline
from multiprocessing import Process
import sys

def run_pipeline():
    pipeline = Pipeline("custom-protocol-pipeline")

    # ... source, decoder, mux, inference, tracker, msgconv setup ...

    # Message converter (converts metadata to message format)
    # IMPORTANT: msg2p-newapi=True uses NvDsObjectMeta directly (no NvDsEventMsgMeta required)
    pipeline.add("nvmsgconv", "msgconv", {
        "config": "msgconv_config.txt",
        "payload-type": 0,  # 0=deepstream full schema, 1=minimal
        "msg2p-newapi": True,  # CRITICAL: Use new API to avoid NvDsEventMsgMeta requirement
    })

    pipeline.add("nvmsgbroker", "msgbroker", {
        "proto-lib": "/path/to/libnvds_custom_proto.so",
        "conn-str": "myserver.example.com;4222",
        "topic": "ds-broker",
        "config": "/path/to/custom_proto_config.txt",
        "sync": 0,
        "async": 0,   # Required when using tee or dynamic sources
        "sleep-time": 10,  # ms between do_work() calls
    })

    pipeline.link("msgconv", "msgbroker")

    try:
        pipeline.start().wait()
    except Exception as e:
        print(f"Pipeline error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    process = Process(target=run_pipeline)
    try:
        process.start()
        process.join()
    except KeyboardInterrupt:
        print("\nInterrupted. Terminating...")
        process.terminate()
        process.join()
```

### Gst-nvmsgbroker Properties for Custom Adapters

| Property | Description |
|----------|-------------|
| `proto-lib` | Absolute path to the custom adapter `.so` file. |
| `conn-str` | Connection string passed to `nvds_msgapi_connect()`. Format defined by the adapter. |
| `config` | Path to the adapter's configuration file, passed to `nvds_msgapi_connect()`. |
| `topic` | Topic name passed to `nvds_msgapi_send()` / `nvds_msgapi_send_async()`. |
| `new-api` | Set to `1` to use the `nv_msgbroker` wrapper library (supports auto-reconnect and connection sharing) instead of calling adapter functions directly. |
| `sleep-time` | Milliseconds between consecutive `nvds_msgapi_do_work()` calls. Default 0. |

### Using new-api (nv_msgbroker Wrapper)

When `new-api=1`, the plugin uses the `nv_msgbroker` wrapper library instead of calling your adapter directly. The wrapper provides:

- **Auto-reconnect**: Periodically retries connection after failures. **IMPORTANT** If this `new-api` property is set to true, the reconnection functionality in the custom adapter library needs to be disabled, as the adapter will conflict with the autoreconnect feature built into `nvmsgbroker`.
- **Connection sharing**: Multiple pipeline components can share a single connection within the same process.
- **Work interval control**: Configurable interval for calling `nvds_msgapi_do_work()`.

The wrapper configuration file (`cfg_nvmsgbroker.txt`):

```ini
[nvmsgbroker]
# Enable auto-reconnection (0=disable, 1=enable)
auto-reconnect=1

# Reconnection retry interval in seconds
retry-interval=1

# Maximum retry limit in seconds
max-retry-limit=3600

# Interval for calling do_work(), in microseconds
work-interval=10000
```

The wrapper internally calls your adapter's `nvds_msgapi_*` functions. Your adapter does not need any changes to work with the wrapper.

---

## Configuration File Format

The config file path is passed as `config_path` to `nvds_msgapi_connect()`. Uses GLib key-file (INI) format. Parse with `GKeyFile` API.

```ini
[message-broker]
server=myserver.example.com
port=5672
username=myuser
share-connection=1
```

```c
/* In nvds_msgapi_connect(), after parsing connection_str: */
GKeyFile *kf = g_key_file_new();
if (config_path && g_key_file_load_from_file(kf, config_path, G_KEY_FILE_NONE, NULL)) {
    gchar *val = g_key_file_get_string(kf, "message-broker", "server", NULL);
    if (val) { ctx->server_url = strdup(val); g_free(val); }
    /* read other keys the same way */
}
g_key_file_free(kf);
```

---

## nvds_logger

Use `nvds_log()` instead of `printf` so logs integrate with DeepStream's log system.

```c
#include "nvds_logger.h"
#define LOG_CAT "DSLOG:CUSTOM_PROTO"  // DSLOG: prefix required for filtering

// In nvds_msgapi_connect(): nvds_log_open();
// In nvds_msgapi_disconnect(): nvds_log_close();

nvds_log(LOG_CAT, LOG_INFO, "Connected to %s:%d", host, port);
nvds_log(LOG_CAT, LOG_ERR,  "Send failed: %d", rc);
nvds_log(LOG_CAT, LOG_DEBUG, "Payload (%zu bytes)", nbuf);
// Link: -L${DS_ROOT}/lib -lnvds_logger
// Logs written to: /tmp/nvds/ds.log
```

---

## Common Pitfalls and Troubleshooting

### Thread Safety and Handle Management

Critical rules for adapter implementations:

1. **Handle lifecycle**: The client (Gst-nvmsgbroker) manages the connection handle lifecycle. Once `nvds_msgapi_disconnect()` is called, the handle is retired and must not be used for send or do_work calls.

2. **Thread safety**: If your underlying protocol library is thread-safe, multiple application threads can share connection handles. If not, you must implement locking or document that handles are single-threaded.

3. **do_work() contract**: If your adapter executes in the client thread (no internal worker thread), you must document how often `nvds_msgapi_do_work()` should be called. If the adapter uses its own threads, `do_work()` can be a no-op.

4. **Graceful failure**: The adapter should attempt graceful failure if called with retired handles, but need not guarantee thread-safe behavior in that case.


### CRITICAL: `nvds_msgapi_send_async()` Callback Must Be Invoked from a Different Thread

**Symptom**: Pipeline deadlocks after the very first message send (or immediately on the first error).

**Root cause** (verified in DeepStream source):

Both code paths lock a mutex **before** calling `nvds_msgapi_send_async()`, and the completion callback tries to re-lock the **same** mutex:

- **Legacy path** (`gstnvmsgbroker.cpp` `legacy_gst_nvmsgbroker_render`): locks `self->flowLock`, calls `nvds_msgapi_send_async()`. The callback `nvds_msgapi_send_callback()` does `g_mutex_lock(&self->flowLock)`.
- **New-api path** (`nvmsgbroker.cpp` `nv_msgbroker_send_async`): locks `h_ptr->do_work_thread.lock`, calls `nvds_msgapi_send_async_ptr()`. The callback `adapter_send_cb()` does `pthread_mutex_lock(&myinfo->h_ptr->do_work_thread.lock)`.

This affects **all** custom adapters, regardless of protocol. Two specific scenarios both deadlock:

- **Success path**: calling `send_callback` synchronously inside `send_async` before returning
- **Error path**: calling `send_callback` with an error code before returning `NVDS_MSGAPI_ERR` — the mutex is still held

**Fix**: Use the **`do_work` thread as the callback boundary** — the pattern used by all five built-in adapters. `send_async()` queues or submits the message and returns immediately without calling the callback. The callback fires later on the do_work thread, which is separate from the render thread that holds the mutex. In the error path, always return `NVDS_MSGAPI_ERR` directly without invoking the callback.

Three proven approaches, in order of applicability:

1. **Event loop + opaque pointer (Pattern B — Kafka)**: pass a heap-allocated `{cb, user_ptr}` as the library's per-message opaque pointer; `do_work()` calls `your_proto_poll()` which fires the delivery callback from within, calling the DeepStream callback there.
2. **Blocking library + atomic queue steal (Pattern D — AMQP, Redis)**: `send_async()` enqueues a `QueueNode` (with copied topic/payload); `do_work()` atomically steals the entire queue, calls the blocking library send for each node, and fires the callback inline. No separate worker thread needed.
3. **Library owns threads (Pattern A — Azure)**: `do_work()` is a no-op; the SDK fires callbacks from its own internal threads, which are already different from the render thread.

Copy topic and payload in all cases where the library does not copy them internally — the caller may free those buffers immediately after `send_async()` returns.

---

## Checklist for Custom Adapter Development

1. **`send_async` callback must run on a different thread** -- see [Common Pitfalls](#common-pitfalls-and-troubleshooting). Both legacy and new-api paths hold a mutex across the `send_async` call; a synchronous callback (including in the error path) deadlocks. Use one of the three proven approaches: event loop + opaque pointer (Pattern B), blocking library + atomic queue steal in `do_work()` (Pattern D), or library-owned threads (Pattern A). Never call the callback before returning from `send_async()`.
2. **Compile as a shared library** (`.so`) with `-shared -fPIC`.
3. **Include `nvds_msgapi.h`** from the DeepStream SDK includes directory.
4. **Handle NULL parameters gracefully** -- `connection_str`, `config_path`, and `topic` may be NULL.
5. **Document the connection string format** your adapter expects.
6. **Document the `do_work()` contract** -- whether the client must call it and how often.
7. **Use `nvds_logger`** for logging instead of raw printf for production adapters.
8. **`deepstream-test5-app`** can be used to test the adapter's deployment and multi-topic subscription functionality.
9. **Test with `gst-launch-1.0`** before integrating with pyservicemaker:

```bash
gst-launch-1.0 \
    filesrc location=test.mp4 ! h264parse ! nvv4l2decoder ! \
    nvstreammux name=mux batch-size=1 width=1920 height=1080 ! \
    nvinfer config-file-path=config.txt ! \
    nvmsgconv config=msgconv_config.txt payload-type=0 msg2p-newapi=1 ! \
    nvmsgbroker proto-lib=/path/to/libnvds_custom_proto.so \
        conn-str="host;port" topic=test config=cfg_custom.txt
```

---

## Related Documentation

- **Kafka Messaging Reference**: `kafka_messaging.md` -- Full Kafka integration patterns and protocol adapter configuration.
- **GStreamer Plugins Overview**: `gstreamer_plugins.md` -- All DeepStream GStreamer plugins including nvmsgbroker and nvmsgconv.
- **Service Maker Python API**: `service_maker_api.md` -- pyservicemaker Pipeline API reference.
