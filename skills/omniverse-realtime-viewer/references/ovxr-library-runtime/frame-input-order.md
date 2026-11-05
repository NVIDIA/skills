# Viewer frame, input and scene scheduling

Read upstream frame-loop, stage-publication, tracking-anchor-and-scale and
openxr-input-and-extensions through [the library route](../dependencies/ovxr.md).
Those contracts own step modes, plan disposal, seal reconciliation and the
frame-latched mapping. The viewer adds the application work below.

## Place application work at the correct boundary

| Boundary | Viewer work |
| --- | --- |
| Before the anchor-latch boundary | Drain bounded lifecycle/navigation intents and ready simulation snapshots; resolve eligible anchor/world-transform publication. Keep runtime waits outside the application scene lock. |
| After predicted timing/mapping is available, before rendering | Validate the input source/epoch/time, normalize one copied snapshot, arbitrate UI versus scene tools, and publish eligible same-frame tool edits through the shared coordinator. |
| After rendering/submission | Correlate applied revisions and submitted-frame evidence. Work computed too late belongs to the next frame. |

The SDK determines where these boundaries fall for the chosen convenience or
split schedule. Do not splice both schedules together or introduce another
renderer/frame owner. Convenience after-frame sampling deliberately feeds a
subsequent frame; label that latency rather than inventing prediction.

## Publication and teardown

Navigation, tools and simulation share the viewer's coordinator. Generation-tag
asynchronous results and reject stale work. Delegate SDK ordinal, cancellation
and release mechanics to upstream; retain the product's requested/applied state
and user-visible error through failed work or context replacement.

Before teardown, quiesce application dispatch, captures, input resources and
mirror/compositor consumers. Validate moving-hand navigation, pending edits,
stale queries and in-flight shutdown using correlated input-to-render evidence.

Before claiming same-frame navigation/input on a selected artifact, test an
anchor change, predicted-time pose conversion and subsequent bind, including
no-render boundaries. If its public timing sources disagree, keep that
capability unverified until the matched-artifact test establishes the effective
mapping; do not compensate with an extra frame loop or mixed timing conventions.
