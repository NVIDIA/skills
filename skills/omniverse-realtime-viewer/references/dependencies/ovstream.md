# ovstream Dependency

## ovstream

Purpose: NVIDIA streaming SDK Python bindings used by browser-streamed Omniverse Realtime Viewers. Do not install it for local-only desktop Omniverse Realtime Viewers.

Read `nvidia-runtime.md` for the latest-version PyPI package guidance. This file
documents runtime setup and validation behavior.

For ovstream-owned skills, sample servers, SHM clients, native input examples,
transport-specific examples, or release-specific behavior, read
`nvidia-runtime.md` for the current ovstream repository pointer and inspect
that repo's `skills/`, samples, examples, and release notes.

## SHM Input Contract

Use the current ovstream client surface for SHM input. The transport is selected
when constructing the client; it is not represented by a separate `ShmClient`
class:

```python
client = ovstream.Client(
    ovstream.ClientType.SHM,
    stream_name="...",
)
client.send_input_event(event)
```

For native C integrations, use the current library symbol:

```c
ovstream_client_send_input_event(client, event);
```

The surrounding input contract remains native `InputEvent` structs. Do not send
JSON `mouseInput` for SHM camera control. For release-specific signatures and
examples, inspect the current ovstream repository `skills/`, examples, headers,
and package docstrings before implementing.


Do not repeat direct wheel URLs, wheel filenames, or alternate package
locations in skills or templates. Keep acquisition details in
`nvidia-runtime.md`.

Set a native library override only if the app's install path cannot locate SDK libraries automatically:

```bash
export OVSTREAM_LIB_PATH=/absolute/path/to/ovstream/native/libs
```

For package retrieval, use the latest-version PyPI package guidance in
`nvidia-runtime.md`. Do not use alternate wheel or tarball locations.
For normal Python apps, the current ovstream wheel is self-contained and
includes the native streaming runtime. Use C/CMake platform zips only for native
C/C++ integrations, or when explicitly debugging extracted layouts.

Verify import and lifecycle:

```bash
python3 -c "import ovstream; ovstream.initialize(); print('ovstream OK', ovstream.get_version()); ovstream.shutdown()"
```

Common failure modes:

- `No matching distribution found for ovstream`: use the PyPI package source in `nvidia-runtime.md` and confirm the latest wheel supports the target OS, architecture, and Python version.
- Package install fails: wrong source, stale package metadata, wrong platform tag, or network/proxy issue.
- Import succeeds but `initialize()` fails with native dependency errors: confirm the installed wheel matches the target OS/architecture and avoid `OVSTREAM_LIB_PATH` overrides unless intentionally debugging an extracted runtime artifact layout.
- Import succeeds but `initialize()` fails for other native errors: runtime artifact does not match OS/architecture, native libraries cannot be found, or driver/GPU support is missing.
- NVENC errors: GPU or driver does not support NVENC, container lacks GPU device access, or another process exhausted encoder resources.
- Browser connects but video never appears: usually app integration, not package install; verify ovstream lifecycle and frame submission through `references/streaming-server`.
