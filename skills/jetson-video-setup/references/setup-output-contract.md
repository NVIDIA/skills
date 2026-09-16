# Setup result

Return a concise human-readable result using this checklist.

For each selected product, report:

- `native` or `pynvc` and one status: `ready`, `installed`, `blocked`, or
  `failed`;
- Jetson Linux release and requested GPU ordinal;
- native package version and package-owned Samples root, or the exact Python
  interpreter, PyNvVideoCodec version, and loaded module path;
- whether a mutation occurred and the exact APT or pip command used;
- official encoder and decoder sample paths;
- observed one-frame markers, bitstream byte count, decoded byte count when a
  raw output is expected, and whether explicit errors were absent; and
- any blocker plus one concrete next action.

Copy every reported path, version, and digest from observed command output;
never retype an identity already captured.

Use `ready` only after the official encoder produces a fresh nonempty
bitstream and the official decoder independently consumes it with the expected
one-frame result. Use `installed` when inventory/import checks pass but the
operation was not requested or not run. A missing prerequisite is `blocked`; a
launched command or validation failure is `failed`.

For a read-only request from another video skill, the compact setup result may
contain only the target/release, selected product, status, and exact local
package/Samples or interpreter/module paths. That consumer must recheck the
paths before its own operation.

Return the result directly in the conversation. Create a file, checksum
manifest, or command-log bundle when the user asks for one.
