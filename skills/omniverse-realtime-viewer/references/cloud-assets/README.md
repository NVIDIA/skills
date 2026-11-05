# Cloud Assets

## Triggers

Use for cloud USD loading, S3-compatible object storage, asset catalogs,
thumbnail browsing, or a local cache of remote assets.

## Architecture

Keep browsing, transfer, and rendering separate:

```text
Asset browser -> storage worker -> prepared local asset
                                      |
                              viewer runtime owner
                                      |
                              OVStage -> OVRTX
```

The UI holds asset identities and load state. The storage worker owns the remote
client and cache. The runtime owner receives a prepared local root path and
performs the scene handoff described in `stage-management`.

Read `references/dependencies/nvidia-runtime.md` for the selected `ovstorage`
package and its upstream integration guidance. Use that SDK's supported
connection, authentication, resolver, and transfer APIs. A public HTTP asset can
use a direct downloader when authentication and resolver integration are not
needed; both paths should expose the same viewer-facing asset identity and load
states.

## Browse Before Downloading

Use a provider's catalog or manifest when available; otherwise page listings by
prefix. Show the first page promptly and load more as the user browses. Preserve
complete object keys as identities and keep display names separate. A repeated
basename in another folder is a different asset.

Load thumbnails on demand with bounded concurrency, and decode/scale them off
the UI thread. A missing thumbnail gets a placeholder; it does not block asset
selection. Cancel superseded browsing work so old results cannot replace the
current folder or search.

## Prepare A Selected Asset

1. Resolve the root USD and its required references, sublayers, textures, and
   materials through the selected resolver or asset manifest. A folder download
   is sufficient only for assets known to be self-contained within it.
2. Fetch required content into a cache preserving the relative layout. Scope
   cache identity to the remote endpoint/bucket and full key. Encode keys using
   the SDK's address helpers rather than concatenating URLs.
3. Validate freshness with provider-supported object versions or validators.
   File size alone cannot detect equal-size updates; use a content comparison
   when reliable metadata is unavailable. Download only changed dependencies.
4. Finish required transfers before publishing the prepared root path. Use
   temporary files and atomic replacement so readers cannot observe partial
   writes. Keep in-use asset generations intact during refresh or eviction.
5. Return the path with the load request's identity. The runtime accepts it only
   if that request is still current, then switches scenes through its normal
   owner-controlled path.

Keep the current scene interactive during transfer. Report loading progress,
allow cancellation, and distinguish download failure from stage-open failure.
A required dependency failure prevents the scene handoff; optional thumbnail
failure does not. On failure, offer retry without silently substituting stale
content for a deleted or inaccessible remote object.

## Storage Adapter Contract

The adapter belongs to the application; these are responsibilities, not SDK
method names:

| Operation | Result |
|---|---|
| List assets | Paged file identities and metadata; distinguish files from inferred directories and reject non-advancing continuation tokens. |
| Prepare asset | Complete dependency closure and a local root path, with progress and cancellation. |
| Refresh | Revalidate selected content without synchronizing the entire bucket on every scene load. |
| Release | Allow eviction after the viewer releases an asset generation. |

For an async SDK client, keep its event loop alive for the client's lifetime.
Run network work in the storage worker, with bounded transfers and timeouts;
stream or materialize large objects instead of reading them all into RAM.
Serialize updates to the same cache entry and reject keys that escape the cache
root or alias another cached path. Delete cached entries only after a complete,
successful refresh establishes that they are obsolete.

## Verify The Integration

Browse a paginated catalog, open an asset with relative dependencies, and reopen
it after a same-size remote change. Check that duplicate basenames stay distinct,
cancelled loads cannot replace a newer selection, required-file failures surface
in the UI, and navigation remains responsive during download.

See also: `stage-management`, `stage-loading`, `viewer-data-view-patterns`,
`viewer-feedback-status`, and `cloud-deployment`.
