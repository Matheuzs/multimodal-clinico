"""Fusão multimodal: combina achados de vídeo (repetições) e áudio (acústica + texto) num risco único.

Regras (todas transparentes e configuráveis em `fusion:` no config.yaml):
1. Cada modalidade pontua a soma, por *regra distinta*, do peso da maior severidade daquela regra
   (10 repetições com o mesmo desvio contam uma vez, não dez).
2. Regras de visão só contam se a fração de repetições sinalizadas ≥ vision_min_flagged_ratio.
3. Bônus de convergência: +convergence_bonus quando visão E áudio têm algum achado ≥ medium.
4. Nível: ALERTA se há achado `high` (high_override) ou se score ≥ alert_threshold COM convergência
   (alert_requires_convergence: várias regras leves de uma só fonte, como vários termos benignos, não bastam);
   ATENÇÃO se score ≥ attention_threshold; senão NORMAL.
Os pesos e limiares são provisórios, sem validação clínica.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.config import load_config
from src.findings import SEVERITY_ORDER, Finding

LEVELS = ("NORMAL", "ATENÇÃO", "ALERTA")


@dataclass
class Evidence:
    modality: str            # "video" | "audio"
    rule: str
    severity: str
    message: str
    detail: str = ""         # ex.: "repetições 2, 4, 5 de 8"


@dataclass
class FusionResult:
    level: str
    score: float
    vision_score: float
    audio_score: float
    convergence: bool
    evidences: list[Evidence] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)
    message: str = ""

    @property
    def is_alert(self) -> bool:
        return self.level == "ALERTA"

    def to_dict(self) -> dict:
        return {
            "level": self.level, "score": self.score, "vision_score": self.vision_score,
            "audio_score": self.audio_score, "convergence": self.convergence,
            "modalities": self.modalities, "message": self.message,
            "evidences": [e.__dict__ for e in self.evidences],
        }


def _top(sevs: list[str]) -> str:
    return max(sevs, key=SEVERITY_ORDER.get)


def _vision_evidence(reps, fc: dict) -> list[Evidence]:
    n = len(reps)
    if not n:
        return []
    by_rule: dict[str, list] = {}
    for r in reps:
        for f in r.findings:
            by_rule.setdefault(f.rule, []).append((r, f))
    out = []
    for rule, items in by_rule.items():
        idx = sorted({r.index for r, _ in items})
        if len(idx) / n < fc["vision_min_flagged_ratio"]:
            continue
        sev = _top([f.severity for _, f in items])
        msg = next(f.message for _, f in items if f.severity == sev)
        out.append(Evidence("video", rule, sev, msg,
                            f"repetições {', '.join(str(i) for i in idx)} de {n}"))
    return out


def _audio_evidence(findings: list[Finding]) -> list[Evidence]:
    by_rule: dict[str, list[Finding]] = {}
    for f in findings:
        by_rule.setdefault(f.rule, []).append(f)
    out = []
    for rule, fs in by_rule.items():
        sev = _top([f.severity for f in fs])
        out.append(Evidence("audio", rule, sev, next(f.message for f in fs if f.severity == sev)))
    return out


def fuse(vision_reps=None, audio_findings: list[Finding] | None = None, cfg: dict | None = None) -> FusionResult:
    """Qualquer modalidade pode ser None (análise de uma só fonte)."""
    cfg = cfg or load_config()
    fc = cfg["fusion"]
    w = fc["severity_weights"]
    ev_v = _vision_evidence(vision_reps, fc) if vision_reps is not None else []
    ev_a = _audio_evidence(audio_findings) if audio_findings is not None else []
    vs = float(sum(w[e.severity] for e in ev_v))
    as_ = float(sum(w[e.severity] for e in ev_a))
    conv = (any(SEVERITY_ORDER[e.severity] >= 2 for e in ev_v)
            and any(SEVERITY_ORDER[e.severity] >= 2 for e in ev_a))
    score = vs + as_ + (fc["convergence_bonus"] if conv else 0)
    evs = sorted(ev_v + ev_a, key=lambda e: -SEVERITY_ORDER[e.severity])
    has_high = any(e.severity == "high" for e in evs)
    score_alert = score >= fc["alert_threshold"] and (conv or not fc["alert_requires_convergence"])
    if score_alert or (fc["high_override"] and has_high):
        level = "ALERTA"
    elif score >= fc["attention_threshold"]:
        level = "ATENÇÃO"
    else:
        level = "NORMAL"
    mods = [m for m, v in (("video", vision_reps), ("audio", audio_findings)) if v is not None]
    res = FusionResult(level, score, vs, as_, conv, evs, mods)
    res.message = build_message(res)
    return res


def build_message(r: FusionResult) -> str:
    if r.level == "NORMAL":
        return "Nenhum desvio relevante identificado nas modalidades analisadas."
    top = "; ".join(f"[{e.modality}] {e.rule}" for e in r.evidences[:3])
    head = ("ALERTA à equipe médica" if r.level == "ALERTA" else "Atenção recomendada")
    conv = " Vídeo e áudio apontam desvios simultâneos." if r.convergence else ""
    return f"{head} (score {r.score:g}). Principais achados: {top}.{conv} Requer avaliação profissional."
