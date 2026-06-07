import argparse
from dataclasses import dataclass


@dataclass
class ViewerConfig:
    width: int = 1920
    height: int = 1080
    target_fps: int = 30
    signaling_port: int = 49100
    http_port: int = 8888
    public_ip: str = "127.0.0.1"
    initial_scene: str = ""
    asset_root: str = ""
    settings_path: str = "data/viewer-settings.json"
    gpu_index: int = 0


def parse_args() -> ViewerConfig:
    p = argparse.ArgumentParser(description="USD Viewer streaming server")
    p.add_argument("--width",           type=int,   default=1920)
    p.add_argument("--height",          type=int,   default=1080)
    p.add_argument("--fps",             type=int,   default=30)
    p.add_argument("--signaling-port",  type=int,   default=49100)
    p.add_argument("--http-port",       type=int,   default=8888)
    p.add_argument("--public-ip",                   default="127.0.0.1")
    p.add_argument("--scene",                       default="")
    p.add_argument("--asset-root",                  default="")
    p.add_argument("--settings-path",               default="data/viewer-settings.json")
    p.add_argument("--gpu",             type=int,   default=0)
    a = p.parse_args()
    return ViewerConfig(
        width=a.width,
        height=a.height,
        target_fps=a.fps,
        signaling_port=a.signaling_port,
        http_port=a.http_port,
        public_ip=a.public_ip,
        initial_scene=a.scene,
        asset_root=a.asset_root,
        settings_path=a.settings_path,
        gpu_index=a.gpu,
    )
