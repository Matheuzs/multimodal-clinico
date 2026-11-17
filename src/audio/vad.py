"""Detecção de fala por energia (RMS em dB), sem dependências além de numpy."""
from __future__ import annotations

import numpy as np

FRAME_S, HOP_S = 0.025, 0.010


def frame_rms_db(y: np.ndarray, sr: int) -> np.ndarray:
    fl, hop = int(FRAME_S * sr), int(HOP_S * sr)
    if len(y) < fl:
        return np.array([-120.0])
    n = 1 + (len(y) - fl) // hop
    idx = np.arange(fl)[None, :] + hop * np.arange(n)[:, None]
    rms = np.sqrt(np.mean(y[idx] ** 2, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-6))


def speech_intervals(y: np.ndarray, sr: int, top_db: float = 30.0,
                     min_gap_s: float = 0.25, min_speech_s: float = 0.15) -> list[tuple[float, float]]:
    """Intervalos (início_s, fim_s) com energia a até `top_db` dB do nível de fala (percentil 95).
    Silêncios menores que `min_gap_s` são fundidos (não contam como pausa)."""
    db = frame_rms_db(y, sr)
    thr = np.percentile(db, 95) - top_db
    active = db > thr
    ivs, start = [], None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            ivs.append([start * HOP_S, i * HOP_S + FRAME_S]); start = None
    if start is not None:
        ivs.append([start * HOP_S, len(active) * HOP_S + FRAME_S])
    merged: list[list[float]] = []
    for a, b in ivs:
        if merged and a - merged[-1][1] < min_gap_s:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= min_speech_s]


def overlap(a: tuple[float, float], intervals: list[tuple[float, float]]) -> float:
    """Segundos de `a` cobertos por `intervals`."""
    return sum(max(0.0, min(a[1], e) - max(a[0], s)) for s, e in intervals)


def frame_f1(pred: list[tuple[float, float]], ref: list[tuple[float, float]], total_s: float) -> dict:
    """Precisão/recall/F1 por frame de 10 ms entre dois conjuntos de intervalos."""
    n = int(total_s / HOP_S) + 1
    def mask(iv):
        m = np.zeros(n, bool)
        for a, b in iv:
            m[int(a / HOP_S): int(b / HOP_S) + 1] = True
        return m
    p, r = mask(pred), mask(ref)
    tp = (p & r).sum()
    prec = tp / max(p.sum(), 1); rec = tp / max(r.sum(), 1)
    return {"precision": float(prec), "recall": float(rec),
            "f1": float(2 * prec * rec / max(prec + rec, 1e-9))}
