"""Segmentação de repetições e regras de desvio para o Ex1 do REHAB24-6
(abdução lateral do braço direito esticado).

Trabalha sobre o JSON gerado por pose_analyzer (série de ângulos por frame).
Avaliação principal RELATIVA: cada repetição é comparada à linha de base da própria pessoa
(mediana das primeiras `baseline_reps`), porque o tipo de erro varia entre pessoas (em uma o
movimento errado é mais lento e curto, em outra mais rápido e amplo). Há também regras absolutas
(cotovelo dobrado, compensação, qualidade). Limiares em configs/config.yaml (vision.exercise),
calibrados em poucos vídeos: indicadores de apoio, não diagnóstico.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.config import load_config

SEVERITY_ORDER = {"low": 1, "medium": 2, "high": 3}


@dataclass
class Finding:
    rule: str
    severity: str          # low | medium | high
    message: str
    value: float
    threshold: float


@dataclass
class Repetition:
    index: int
    start_frame: int
    end_frame: int
    peak_frame: int
    features: dict
    findings: list[Finding] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return bool(self.findings)

    @property
    def severity(self) -> str | None:
        if not self.findings:
            return None
        return max((f.severity for f in self.findings), key=SEVERITY_ORDER.get)


def _series(frames: list[dict], key: str):
    """Retorna (frames, valores) interpolando linearmente lacunas (ângulo None)."""
    idx = np.array([f["frame"] for f in frames], float)
    val = np.array([f["angles"].get(key) if f["angles"].get(key) is not None else np.nan
                    for f in frames], float)
    ok = ~np.isnan(val)
    if ok.sum() < 2:
        return idx, np.full_like(val, np.nan)
    return idx, np.interp(idx, idx[ok], val[ok])


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    pad = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge")
    return np.convolve(pad, np.ones(w) / w, mode="valid")


def segment_repetitions(pose: dict, cfg: dict | None = None) -> list[tuple[int, int, int]]:
    """Histerese no ângulo do ombro direito. Devolve [(início, fim, frame_do_pico)]."""
    ex = (cfg or load_config())["vision"]["exercise"]
    frames = pose["frames"]
    idx, sh = _series(frames, "right_shoulder")
    if np.isnan(sh).all():
        return []
    sh = _smooth(sh, ex["smooth_window"])
    # Repetições truncadas (vídeo começa com o braço já levantado, ou termina no meio do
    # movimento) são descartadas: não dá para avaliar amplitude de um movimento incompleto.
    reps, start, in_rep = [], None, False
    for i, v in enumerate(sh):
        if not in_rep and v >= ex["rep_start_deg"]:
            in_rep, start = True, i
        elif in_rep and v < ex["rep_end_deg"]:
            seg = slice(start, i + 1)
            if start > 0 and sh[seg].max() >= ex["rep_min_peak_deg"]:
                reps.append((int(idx[start]), int(idx[i]), int(idx[start + int(np.argmax(sh[seg]))])))
            in_rep = False
    return reps


def extract_features(pose: dict, start: int, end: int, fps: float) -> dict:
    fr = [f for f in pose["frames"] if start <= f["frame"] <= end]
    out = {"duration_s": round((end - start) / fps, 2), "n_frames": len(fr),
           "detected_ratio": round(sum(f["detected"] for f in fr) / max(len(fr), 1), 3)}
    for name, key, fn in [("peak_shoulder", "right_shoulder", np.nanmax),
                          ("min_elbow", "right_elbow", np.nanmin)]:
        _, v = _series(fr, key)
        out[name] = round(float(fn(v)), 1) if len(v) and not np.isnan(v).all() else None
    idx, sh = _series(fr, "right_shoulder")
    if len(sh) >= 3 and not np.isnan(sh).any():
        out["speed_up"] = round(float(np.max(np.gradient(_smooth(sh, 3), idx / fps))), 1)  # graus/s
    else:
        out["speed_up"] = None
    _, ls = _series(fr, "left_shoulder")
    out["other_shoulder_range"] = (round(float(np.nanmax(ls) - np.nanmin(ls)), 1)
                                   if len(ls) and not np.isnan(ls).all() else None)
    return out


def _band(dev: float, thr: float) -> str:
    r = abs(dev) / thr
    return "low" if r <= 1.5 else "medium" if r <= 2.5 else "high"


def compute_baseline(features: list[dict], cfg: dict | None = None) -> dict | None:
    """Mediana das primeiras `baseline_reps` repetições. None se houver menos de 3."""
    k = (cfg or load_config())["vision"]["exercise"]["baseline_reps"]
    ref = features[:k]
    if len(ref) < 3:
        return None
    base = {}
    for key in ("peak_shoulder", "speed_up", "min_elbow"):
        vals = [f[key] for f in ref if f.get(key) is not None]
        base[key] = float(np.median(vals)) if vals else None
    return base


def evaluate_rules(f: dict, cfg: dict | None = None, baseline: dict | None = None) -> list[Finding]:
    ex = (cfg or load_config())["vision"]["exercise"]
    out: list[Finding] = []

    # --- regras relativas (linha de base da própria pessoa)
    if baseline:
        b = baseline.get("peak_shoulder")
        if b and f["peak_shoulder"] is not None:
            dev = 100 * (f["peak_shoulder"] - b) / b
            if abs(dev) > ex["rel_peak_pct"]:
                sentido = "maior" if dev > 0 else "menor"
                out.append(Finding("desvio_amplitude", _band(dev, ex["rel_peak_pct"]),
                                   f"Amplitude {sentido} que a sua referência: {f['peak_shoulder']:.0f}° "
                                   f"({dev:+.0f}% vs {b:.0f}°)", f["peak_shoulder"], b))
        b = baseline.get("speed_up")
        if b and f.get("speed_up") is not None:
            dev = 100 * (f["speed_up"] - b) / b
            if abs(dev) > ex["rel_speed_pct"]:
                sentido = "mais rápido" if dev > 0 else "mais lento"
                out.append(Finding("desvio_velocidade", _band(dev, ex["rel_speed_pct"]),
                                   f"Movimento {sentido} que a sua referência: {f['speed_up']:.0f}°/s "
                                   f"({dev:+.0f}% vs {b:.0f}°/s)", f["speed_up"], b))
        b = baseline.get("min_elbow")
        if b and f["min_elbow"] is not None:
            dev = f["min_elbow"] - b
            if abs(dev) > ex["rel_elbow_deg"]:
                sentido = "mais dobrado" if dev < 0 else "mais esticado"
                out.append(Finding("desvio_cotovelo", _band(dev, ex["rel_elbow_deg"]),
                                   f"Cotovelo {sentido} que a sua referência: {f['min_elbow']:.0f}° "
                                   f"({dev:+.0f}° vs {b:.0f}°)", f["min_elbow"], b))

    # --- regras absolutas
    min_peak = ex.get("min_peak_shoulder_deg")
    if min_peak is not None and f["peak_shoulder"] is not None and f["peak_shoulder"] < min_peak:
        deficit = min_peak - f["peak_shoulder"]
        sev = "low" if deficit <= 5 else "medium" if deficit <= 15 else "high"
        out.append(Finding("amplitude", sev,
                           f"Amplitude insuficiente: braço chegou a {f['peak_shoulder']:.0f}° "
                           f"(mínimo esperado {min_peak}°)", f["peak_shoulder"], min_peak))
    if f["min_elbow"] is not None and f["min_elbow"] < ex["min_elbow_deg"]:
        deficit = ex["min_elbow_deg"] - f["min_elbow"]
        sev = "low" if deficit <= 10 else "medium" if deficit <= 25 else "high"
        out.append(Finding("cotovelo_dobrado", sev,
                           f"Cotovelo dobrado ({f['min_elbow']:.0f}°); o braço deve ficar esticado",
                           f["min_elbow"], ex["min_elbow_deg"]))
    if f["other_shoulder_range"] is not None and f["other_shoulder_range"] > ex["max_other_shoulder_range_deg"]:
        out.append(Finding("compensacao", "medium",
                           f"Ombro oposto se moveu {f['other_shoulder_range']:.0f}° (possível compensação do tronco)",
                           f["other_shoulder_range"], ex["max_other_shoulder_range_deg"]))
    if f["detected_ratio"] < ex["min_detected_ratio"]:
        out.append(Finding("qualidade", "low",
                           f"Pose detectada em só {100 * f['detected_ratio']:.0f}% dos frames; resultado pouco confiável",
                           f["detected_ratio"], ex["min_detected_ratio"]))
    return out


def analyze_repetitions(pose: dict, cfg: dict | None = None) -> list[Repetition]:
    cfg = cfg or load_config()
    segs = segment_repetitions(pose, cfg)
    feats = [extract_features(pose, s, e, pose["fps"]) for s, e, _ in segs]
    baseline = compute_baseline(feats, cfg)
    return [Repetition(i, s, e, p, f, evaluate_rules(f, cfg, baseline))
            for i, ((s, e, p), f) in enumerate(zip(segs, feats), 1)]
