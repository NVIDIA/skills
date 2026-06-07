import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import { AppStreamer } from "@nvidia/ov-web-rtc";
import { getStreamingConfig } from "./streamingConfig";
import { MSG } from "./messages";
import type {
  DataMessage,
  CameraState,
  RenderSettings,
  Hierarchy,
} from "../types/messages";

type StreamStatus = "connecting" | "connected" | "disconnected" | "error";

interface StreamingState {
  status:         StreamStatus;
  cameraState:    CameraState    | null;
  renderSettings: RenderSettings | null;
  hierarchy:      Hierarchy      | null;
  lastError:      string         | null;
  videoRef:       React.RefObject<HTMLVideoElement>;
  send:           (msg: string) => void;
}

const Ctx = createContext<StreamingState | null>(null);

export function useStreaming(): StreamingState {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useStreaming must be inside StreamingProvider");
  return ctx;
}

export function StreamingProvider({ children }: { children: React.ReactNode }) {
  const videoRef      = useRef<HTMLVideoElement>(null);
  const streamerRef   = useRef<AppStreamer | null>(null);
  const [status,         setStatus]         = useState<StreamStatus>("disconnected");
  const [cameraState,    setCameraState]    = useState<CameraState    | null>(null);
  const [renderSettings, setRenderSettings] = useState<RenderSettings | null>(null);
  const [hierarchy,      setHierarchy]      = useState<Hierarchy      | null>(null);
  const [lastError,      setLastError]      = useState<string         | null>(null);

  const send = useCallback((msg: string) => {
    streamerRef.current?.sendMessage(msg);
  }, []);

  const onMessage = useCallback((raw: string) => {
    let msg: DataMessage;
    try { msg = JSON.parse(raw); } catch { return; }
    switch (msg.event_type) {
      case "cameraState":    setCameraState(msg.payload    as CameraState);    break;
      case "renderSettings": setRenderSettings(msg.payload as RenderSettings); break;
      case "hierarchy":      setHierarchy(msg.payload      as Hierarchy);      break;
      case "stageLoaded":    send(MSG.getHierarchy());                          break;
      case "error":          setLastError((msg.payload as { message: string }).message); break;
    }
  }, [send]);

  useEffect(() => {
    const cfg = getStreamingConfig();
    const streamer = new AppStreamer({
      server:        cfg.serverHost,
      signalingPort: cfg.signalingPort,
      videoElement:  videoRef.current!,
      onOpen:    () => { setStatus("connected"); setLastError(null); },
      onClose:   () => setStatus("disconnected"),
      onError:   (e: Error) => { setStatus("error"); setLastError(e.message); },
      onMessage,
    });
    streamer.connect();
    streamerRef.current = streamer;
    setStatus("connecting");
    return () => { streamer.disconnect(); streamerRef.current = null; };
  }, [onMessage]);

  return (
    <Ctx.Provider value={{
      status, cameraState, renderSettings, hierarchy, lastError, videoRef, send,
    }}>
      {children}
    </Ctx.Provider>
  );
}
