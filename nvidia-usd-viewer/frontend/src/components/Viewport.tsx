import React from "react";
import { useStreaming } from "../streaming/StreamingProvider";

const OVERLAY: React.CSSProperties = {
  position:       "absolute",
  inset:          0,
  display:        "flex",
  flexDirection:  "column",
  alignItems:     "center",
  justifyContent: "center",
  color:          "#888",
  fontSize:       14,
  gap:            8,
  pointerEvents:  "none",
};

export function Viewport() {
  const { videoRef, status } = useStreaming();
  return (
    <div style={{
      flex: 1, background: "#111",
      display: "flex", alignItems: "center", justifyContent: "center",
      position: "relative", overflow: "hidden",
    }}>
      {/* Browser displays <video> only — no Three.js / WebGL / canvas 3D */}
      <video
        ref={videoRef}
        autoPlay playsInline muted
        style={{ width: "100%", height: "100%", objectFit: "contain", display: "block" }}
      />
      {status !== "connected" && (
        <div style={OVERLAY}>
          <span style={{ fontSize: 28 }}>◌</span>
          <span>{status === "connecting" ? "Connecting to RTX renderer…" : "Not connected"}</span>
        </div>
      )}
    </div>
  );
}
