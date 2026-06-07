import React, { useState } from "react";
import { useStreaming } from "../streaming/StreamingProvider";
import { MSG } from "../streaming/messages";

export function Toolbar() {
  const { send, status } = useStreaming();
  const [url, setUrl]    = useState("");
  const canLoad          = status === "connected" && url.trim().length > 0;

  const load = () => { if (canLoad) send(MSG.openStage(url.trim())); };

  return (
    <div style={{
      height: 48, background: "#1a1a2e",
      borderBottom: "1px solid #333",
      display: "flex", alignItems: "center",
      padding: "0 16px", gap: 12, flexShrink: 0,
    }}>
      <span style={{ color: "#76b900", fontWeight: 700, fontSize: 14, marginRight: 4 }}>
        USD Viewer
      </span>
      <input
        value={url}
        onChange={e => setUrl(e.target.value)}
        onKeyDown={e => e.key === "Enter" && load()}
        placeholder="USD scene path or URL…"
        style={{
          flex: 1, maxWidth: 480,
          background: "#0d0d1a", border: "1px solid #444", borderRadius: 4,
          padding: "6px 10px", color: "#eee", fontSize: 13, outline: "none",
        }}
      />
      <button
        onClick={load}
        disabled={!canLoad}
        style={{
          background: canLoad ? "#76b900" : "#333",
          color: canLoad ? "#fff" : "#666",
          border: "none", borderRadius: 4,
          padding: "6px 14px", fontSize: 13,
          cursor: canLoad ? "pointer" : "not-allowed",
        }}
      >
        Load
      </button>
    </div>
  );
}
