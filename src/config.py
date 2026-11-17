"""Carrega configs/config.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict:
    path = path or ROOT / "configs" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

