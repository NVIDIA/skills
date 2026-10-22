#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Open the Isaac Sim WebRTC viewer in Chromium and send its Play shortcut.

Usage:
    python3 play_viewer.py --url URL [--browser PATH] [--timeout SECONDS]

Arguments:
    --url is required; browser, debug port, timeout, and ready delay are optional.

Output:
    Prints confirmation after the focused viewer receives the Play shortcut.

Exit codes:
    0 means the shortcut was sent; 1 means browser or WebRTC control failed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import tempfile

from http_transport import open_http

import websocket


def cdp_command(socket: websocket.WebSocket, message_id: int, method: str, params: dict) -> None:
    socket.send(json.dumps({"id": message_id, "method": method, "params": params}))
    while True:
        response = json.loads(socket.recv())
        if response.get("id") == message_id:
            if "error" in response:
                raise RuntimeError(f"{method} failed: {response['error']}")
            return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--browser", default="chromium")
    parser.add_argument("--debug-port", type=int, default=9222)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument(
        "--viewer-ready-delay", type=float, default=5.0,
        help="Seconds to wait for the WebRTC session before sending Play",
    )
    args = parser.parse_args()

    if not 1 <= args.debug_port <= 65535:
        parser.error("--debug-port must be between 1 and 65535")

    with tempfile.TemporaryDirectory(prefix="isaac-sim-viewer-") as profile:
        return play(args, profile)


def play(args, profile):
    browser = subprocess.Popen(
        [
            args.browser,
            "--headless=new",
            "--autoplay-policy=no-user-gesture-required",
            f"--remote-debugging-port={args.debug_port}",
            "--remote-allow-origins=*",
            f"--user-data-dir={profile}",
            args.url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    try:
        endpoint = f"http://127.0.0.1:{args.debug_port}/json/list"
        deadline = time.monotonic() + args.timeout
        page = None
        while time.monotonic() < deadline:
            try:
                with open_http(endpoint, timeout=2) as response:
                    targets = json.load(response)
                page = next(
                    (target for target in targets if target.get("type") == "page" and args.url in target.get("url", "")),
                    None,
                )
                if page:
                    break
            except OSError:
                pass
            time.sleep(0.25)
        if not page:
            raise RuntimeError(f"Chromium did not open the viewer within {args.timeout}s")

        time.sleep(args.viewer_ready_delay)
        socket = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=args.timeout)
        try:
            cdp_command(socket, 1, "Page.bringToFront", {})
            cdp_command(socket, 2, "Input.dispatchMouseEvent", {"type": "mousePressed", "x": 640, "y": 360, "button": "left", "clickCount": 1})
            cdp_command(socket, 3, "Input.dispatchMouseEvent", {"type": "mouseReleased", "x": 640, "y": 360, "button": "left", "clickCount": 1})
            key = {"key": " ", "code": "Space", "windowsVirtualKeyCode": 32, "nativeVirtualKeyCode": 32}
            cdp_command(socket, 4, "Input.dispatchKeyEvent", {"type": "keyDown", **key})
            cdp_command(socket, 5, "Input.dispatchKeyEvent", {"type": "keyUp", **key})
        finally:
            socket.close()
    finally:
        browser.terminate()
        try:
            browser.wait(timeout=3)
        except subprocess.TimeoutExpired:
            browser.kill()
            browser.wait()
    print("Play shortcut sent through Chromium to the focused WebRTC viewer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
