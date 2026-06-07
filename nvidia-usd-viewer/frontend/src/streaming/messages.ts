import type { EventType } from "../types/messages";

function msg<T>(event_type: EventType, payload: T): string {
  return JSON.stringify({ event_type, payload });
}

export const MSG = {
  openStage:   (url: string)    => msg("openStageRequest",  { url }),
  getCamera:   ()               => msg("getCameraState",    {}),
  getHierarchy:()               => msg("getHierarchy",      {}),
  getSettings: ()               => msg("getRenderSettings", {}),
  setSettings: (s: object)      => msg("setRenderSettings", s),
  ping:        ()               => msg("ping",              {}),
};
