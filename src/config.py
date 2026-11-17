"""Carrega configs/config.yaml e variáveis de ambiente (.env)."""
from __future__ import annotations

from pathlib import Path
import os

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict:
    load_dotenv(ROOT / ".env")
    path = path or ROOT / "configs" / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

