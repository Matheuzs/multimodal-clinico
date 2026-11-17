"""Achado clínico genérico, compartilhado por visão, áudio e fusão."""
from __future__ import annotations

from dataclasses import dataclass

SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3}


@dataclass
class Finding:
    rule: str
    severity: str          # low | medium | high
    message: str
    value: float
    threshold: float
