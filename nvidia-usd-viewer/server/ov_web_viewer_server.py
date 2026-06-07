"""USD Viewer streaming server — entry point."""
from __future__ import annotations

import json
import logging
import os
import queue
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

os.environ.setdefault("OVRTX_SKIP_USD_CHECK", "1")

from server.camera_controller import CameraController
from server.config            import parse_args
from server.message_router    import MessageRouter
from server.input_router      import InputRouter
from server.renderer_runtime  import RendererRuntime
from server.scene_loader      import SceneLoader
from server.settings_store    import SettingsStore
from server.stage_queries     import get_hierarchy
from server.stream_server     import StreamServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
log = logging.getLogger("usd_viewer")


class _HealthServer:
    def __init__(self, port: int) -> None:
        self._port  = port
        self._ready = False
        self._srv: Optional[HTTPServer] = None

    def set_ready(self) -> None:
        self._ready = True
        log.info("/healthz → 200 ready")

    def start(self) -> None:
        outer = self

        class _H(BaseHTTPRequestHandler):
            def do_GET(self):
                code, body = (200, b"ok") if outer._ready else (503, b"not ready")
                self.send_response(code)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_):
                pass

        self._srv = HTTPServer(("0.0.0.0", self._port), _H)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        log.info("Health server  :%d", self._port)

    def stop(self) -> None:
        if self._srv:
            self._srv.shutdown()


class ViewerApp:
    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._q: queue.Queue = queue.Queue()
        self._running = False

        self._renderer = RendererRuntime(cfg.width, cfg.height, cfg.gpu_index)
        self._loader   = SceneLoader(cfg.width, cfg.height, cfg.asset_root)
        self._settings = SettingsStore(cfg.settings_path)
        self._health   = _HealthServer(cfg.http_port)

        self._camera = CameraController(
            cfg.width, cfg.height,
            write_fn=lambda mat: self._renderer.write_camera_transform(mat),
        )

        self._stream = StreamServer(
            width=cfg.width, height=cfg.height,
            target_fps=cfg.target_fps,
            signaling_port=cfg.signaling_port,
            public_ip=cfg.public_ip,
            on_connection=self._on_connection,
            on_message=lambda msg:  self._q.put(("msg",    msg)),
            on_input=lambda data:   self._q.put(("input",  data)),
        )

        self._router = MessageRouter(
            send_fn=self._stream.send_message,
            load_scene_fn=lambda url: self._q.put(("load", url)),
            get_camera_fn=self._camera.get_state,
            set_camera_fn=lambda s:  self._q.put(("set_cam", s)),
            get_hierarchy_fn=lambda: get_hierarchy(self._renderer.native_renderer),
            get_settings_fn=self._settings.get,
            set_settings_fn=lambda p: self._settings.update(p),
        )

        self._input = InputRouter(
            on_mouse_button=lambda *a: self._q.put(("btn",    a)),
            on_mouse_move=lambda *a:   self._q.put(("move",   a)),
            on_mouse_scroll=lambda *a: self._q.put(("scroll", a)),
            stream_width=cfg.width,
            stream_height=cfg.height,
        )

    def run(self) -> None:
        log.info("USD Viewer  %dx%d @ %dfps  GPU=%d",
                 self._cfg.width, self._cfg.height,
                 self._cfg.target_fps, self._cfg.gpu_index)

        self._renderer.initialize(target_fps=self._cfg.target_fps)

        if self._cfg.initial_scene:
            self._load_scene(self._cfg.initial_scene)

        self._renderer.warmup(frames=8)

        self._stream.initialize()
        self._stream.start()
        self._health.start()

        self._running = True
        signal.signal(signal.SIGINT,  self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        self._render_loop()

    def _load_scene(self, url: str) -> None:
        usda, rp_path, cam_path = self._loader.build_usda(url)
        self._renderer.open_usd_from_string(usda, rp_path, cam_path)
        self._router.send_scene_loaded(url)

    def _render_loop(self) -> None:
        frame_dt  = 1.0 / self._cfg.target_fps
        first_frame = True

        while self._running:
            t0 = time.monotonic()

            while not self._q.empty():
                try:
                    self._dispatch(self._q.get_nowait())
                except queue.Empty:
                    break

            bgra = self._renderer.step_and_get_bgra()
            if bgra is not None:
                self._stream.stream_video(bgra)
                if first_frame:
                    self._health.set_ready()
                    first_frame = False

            sleep = frame_dt - (time.monotonic() - t0)
            if sleep > 0:
                time.sleep(sleep)

        self._shutdown()

    def _dispatch(self, cmd: tuple) -> None:
        kind = cmd[0]
        if   kind == "load":    self._load_scene(cmd[1])
        elif kind == "msg":     self._router.handle(cmd[1])
        elif kind == "input":   self._input.handle(cmd[1])
        elif kind == "btn":     self._camera.on_mouse_button(*cmd[1])
        elif kind == "move":    self._camera.on_mouse_move(*cmd[1])
        elif kind == "scroll":  self._camera.on_mouse_scroll(*cmd[1])
        elif kind == "set_cam": self._camera.set_state(cmd[1])

    def _on_connection(self, connected: bool) -> None:
        if connected:
            self._router.send_initial_state()

    def _on_signal(self, sig, _frame) -> None:
        log.info("Signal %d — shutting down", sig)
        self._running = False

    def _shutdown(self) -> None:
        self._stream.stop()
        self._renderer.shutdown()
        self._health.stop()
        log.info("Shutdown complete")


def main():
    cfg = parse_args()
    ViewerApp(cfg).run()


if __name__ == "__main__":
    main()
