"""Métricas acústicas da fala do paciente: pausas, ritmo de fala e energia.

São indicadores DESCRITIVOS (sem rótulo clínico no PriMock57). Pausas só contam quando o paciente
está "de posse da palavra": se o canal do médico é informado, silêncios cobertos pela fala dele
(>= 50 %) são troca de turno; sem ele, silêncios maiores que `turn_gap_s` são descartados.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.audio.transcriber import read_audio_16k
from src.audio.vad import HOP_S, frame_rms_db, overlap, speech_intervals
from src.config import load_config
from src.findings import Finding

SR = 16000


@dataclass
class AcousticFeatures:
    duration_s: float
    speech_s: float                 # tempo com fala do paciente
    active_s: float                 # fala + pausas do paciente (descontadas as trocas de turno)
    n_pauses: int
    n_long_pauses: int
    pause_s: float
    pause_ratio: float              # pausas / (fala + pausas)
    max_pause_s: float
    long_pauses_per_min: float      # por minuto de tempo ativo
    rms_db_mean: float              # nível relativo (dBFS) nos trechos de fala
    energy_trend_db: float | None   # último terço - primeiro terço da fala (dB)
    n_words: int | None = None
    speech_rate_wps: float | None = None   # palavras por segundo de fala (sem pausas)

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _pauses(speech: list[tuple[float, float]], other: list[tuple[float, float]] | None,
            turn_gap_s: float) -> list[float]:
    out = []
    for (_, e1), (s2, _) in zip(speech, speech[1:]):
        gap = (e1, s2)
        g = s2 - e1
        if other is not None:
            if overlap(gap, other) >= 0.5 * g:
                continue
        elif g > turn_gap_s:
            continue
        out.append(g)
    return out


def extract_features(wav_path: str | Path, n_words: int | None = None,
                     other_wav_path: str | Path | None = None, cfg: dict | None = None) -> AcousticFeatures:
    cfg = cfg or load_config()
    a = cfg["audio"]
    v = a["vad"]
    long_s = a["thresholds"]["long_pause_s"]
    y = read_audio_16k(wav_path)
    dur = len(y) / SR
    speech = speech_intervals(y, SR, v["top_db"], v["min_gap_s"])
    other = None
    if other_wav_path is not None:
        other = speech_intervals(read_audio_16k(other_wav_path), SR, v["top_db"], v["min_gap_s"])

    speech_s = sum(b - s for s, b in speech)
    pauses = _pauses(speech, other, v["turn_gap_s"])
    pause_s = float(sum(pauses))
    active = speech_s + pause_s
    long_n = sum(p >= long_s for p in pauses)

    db = frame_rms_db(y, SR)
    mask = np.zeros(len(db), bool)
    for s, b in speech:
        mask[int(s / HOP_S): int(b / HOP_S) + 1] = True
    sp_db = db[mask] if mask.any() else db
    trend = None
    if mask.sum() >= 300:                       # >= 3 s de fala
        t = np.array_split(sp_db, 3)
        trend = float(t[2].mean() - t[0].mean())

    return AcousticFeatures(
        duration_s=dur, speech_s=speech_s, active_s=active, n_pauses=len(pauses),
        n_long_pauses=long_n, pause_s=pause_s, pause_ratio=pause_s / active if active else 0.0,
        max_pause_s=max(pauses) if pauses else 0.0,
        long_pauses_per_min=long_n / (active / 60) if active else 0.0,
        rms_db_mean=float(sp_db.mean()), energy_trend_db=trend,
        n_words=n_words, speech_rate_wps=(n_words / speech_s if n_words and speech_s else None))


def _band(ratio: float) -> str:
    return "low" if ratio <= 1.5 else "medium" if ratio <= 2.5 else "high"


def evaluate_acoustic(f: AcousticFeatures, cfg: dict | None = None) -> list[Finding]:
    """Indicadores de possível fadiga/dificuldade para falar. Descritivos, não diagnósticos."""
    t = (cfg or load_config())["audio"]["thresholds"]
    out: list[Finding] = []
    if f.speech_rate_wps is not None and f.speech_rate_wps < t["low_speech_rate_wps"]:
        out.append(Finding("fala_lenta", _band(t["low_speech_rate_wps"] / f.speech_rate_wps),
                           f"Fala lenta: {f.speech_rate_wps:.1f} palavras/s (referência > {t['low_speech_rate_wps']})",
                           f.speech_rate_wps, t["low_speech_rate_wps"]))
    if f.long_pauses_per_min > t["max_long_pauses_per_min"]:
        out.append(Finding("pausas_longas", _band(f.long_pauses_per_min / t["max_long_pauses_per_min"]),
                           f"{f.n_long_pauses} pausas longas (>= {t['long_pause_s']} s), "
                           f"{f.long_pauses_per_min:.1f} por minuto",
                           f.long_pauses_per_min, t["max_long_pauses_per_min"]))
    if f.pause_ratio > t["max_pause_ratio"]:
        out.append(Finding("muita_pausa", _band(f.pause_ratio / t["max_pause_ratio"]),
                           f"{100 * f.pause_ratio:.0f}% do tempo de fala em pausas",
                           f.pause_ratio, t["max_pause_ratio"]))
    if f.energy_trend_db is not None and f.energy_trend_db < -t["energy_drop_db"]:
        out.append(Finding("queda_energia", _band(-f.energy_trend_db / t["energy_drop_db"]),
                           f"Voz perdeu {-f.energy_trend_db:.0f} dB de energia ao longo da fala (possível cansaço)",
                           f.energy_trend_db, -t["energy_drop_db"]))
    return out
