# Passthrough in an XR viewer

Read [library session-modes-and-passthrough](../dependencies/ovxr.md) for requested versus negotiated blending and the selected runtime/client's separate passthrough contract.

Compose renderer alpha, OVXR submission and client presentation. [AR compositing](../ovxr-library-runtime/ar-compositing.md) diagnoses the producing image; black desktop pixels cannot establish transparency.

Keep background visibility, illumination, glass/cutouts, matte receivers and exposure independent. [AR controls](../ovxr-library-runtime/live-ar-controls.md) use only supported public apply paths; arbitrary successful attribute writes do not prove live support.

Expose saved policy and effective state. Mode changes follow supported context/service lanes; renderer controls do not renegotiate opaque sessions. Test material/alpha fixtures on the intended client with correlated frames and continued head motion.
