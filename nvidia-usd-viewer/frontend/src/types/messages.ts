export type EventType =
  | "openStageRequest"
  | "getCameraState"
  | "setCameraState"
  | "getHierarchy"
  | "getRenderSettings"
  | "setRenderSettings"
  | "ping"
  | "pong"
  | "stageLoaded"
  | "cameraState"
  | "hierarchy"
  | "renderSettings"
  | "error";

export interface DataMessage<T = unknown> {
  event_type: EventType;
  payload: T;
}

export interface CameraState {
  target:    [number, number, number];
  distance:  number;
  azimuth:   number;
  elevation: number;
}

export interface RenderSettings {
  renderPreset:    "quality" | "realtime";
  samplesPerPixel: number;
  maxBounces:      number;
  toneMapping:     string;
  exposure:        number;
}

export interface PrimInfo {
  path: string;
  type: string;
}

export interface Hierarchy {
  prims: PrimInfo[];
}
