"""Persists viewer settings to a JSON file across sessions."""
from __future__ import annotations

import json
import logging
import os

log = logging.getLogger(__name__)

_DEFAULTS = {
    "renderPreset":    "quality",
    "samplesPerPixel": 64,
    "maxBounces":      6,
    "toneMapping":     "aces",
    "exposure":        0.0,
}


class SettingsStore:
    def __init__(self, path: str = "data/viewer-settings.json") -> None:
        self._path = path
        self._data = dict(_DEFAULTS)
        self._load()

    def get(self) -> dict:
        return dict(self._data)

    def update(self, patch: dict) -> None:
        self._data.update(patch)
        self._save()

    def _load(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    self._data.update(json.load(f))
            except Exception as exc:
                log.warning("Could not load settings: %s", exc)

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        try:
            with open(self._path, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception as exc:
            log.warning("Could not save settings: %s", exc)
