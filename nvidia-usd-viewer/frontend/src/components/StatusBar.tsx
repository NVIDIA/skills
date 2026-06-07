import React from "react";
import { useStreaming } from "../streaming/StreamingProvider";

const LABELS: Record<string, string> = {
  connecting:   "Connecting…",
  connected:    "Streaming",
  disconnected: "Disconnected",
  error:        "Error",
};
const COLORS: Record<string, string> = {
  connecting:   "#f0a500",
  connected:    "#76b900",
  disconnected: "#666",
  error:        "#e53935",
};

export function StatusBar() {
  const { status, lastError } = useStreaming();
  return (
    <div style={{
      height: 24, background: "#111",
      borderTop: "1px solid #222",
      display: "flex", alignItems: "center",
      padding: "0 16px", gap: 8, flexShrink: 0,
    }}>
      <span style={{
        width: 8, height: 8, borderRadius: "50%",
        background: COLORS[status] ?? "#666",
        display: "inline-block",
      }} />
      <span style={{ color: "#aaa", fontSize: 11 }}>
        {LABELS[status] ?? status}
        {lastError ? ` — ${lastError}` : ""}
      </span>
      <span style={{ marginLeft: "auto", color: "#444", fontSize: 11 }}>
        RTX · ovrtx + ovstream
      </span>
    </div>
  );
}
