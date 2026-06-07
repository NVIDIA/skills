import React from "react";
import { StreamingProvider } from "./streaming/StreamingProvider";
import { Viewport }   from "./components/Viewport";
import { Toolbar }    from "./components/Toolbar";
import { StatusBar }  from "./components/StatusBar";

export default function App() {
  return (
    <StreamingProvider>
      <div style={{
        display: "flex", flexDirection: "column",
        height: "100vh", width: "100vw",
        background: "#0d0d1a",
        fontFamily: "system-ui, sans-serif",
      }}>
        <Toolbar />
        <Viewport />
        <StatusBar />
      </div>
    </StreamingProvider>
  );
}
