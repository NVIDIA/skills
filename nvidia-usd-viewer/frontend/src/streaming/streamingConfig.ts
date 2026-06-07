export interface StreamingConfig {
  serverHost:    string;
  signalingPort: number;
}

export function getStreamingConfig(): StreamingConfig {
  const p = new URLSearchParams(window.location.search);
  return {
    serverHost:    p.get("host") ?? "127.0.0.1",
    signalingPort: Number(p.get("port") ?? "49100"),
  };
}
