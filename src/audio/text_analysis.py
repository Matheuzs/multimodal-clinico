"""Termos críticos (com negação) e sentimento sobre a transcrição do paciente.

Termos: dicionário em configs/config.yaml (audio.critical_terms). Um termo precedido de negação
na mesma frase ("no fever", "denies chest pain") é registrado como negado e NÃO gera alerta.
Sentimento: pipeline('sentiment-analysis') do Transformers (aula 06) ou um léxico simples (`mock`).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from src.config import load_config, use_azure
from src.findings import SEVERITY_ORDER, Finding

NEGATORS = {"no", "not", "never", "without", "denies", "denied", "deny", "none", "nothing", "nor",
            "neither", "haven't", "hasn't", "don't", "doesn't", "didn't", "isn't", "wasn't",
            "aren't", "weren't", "havent", "dont", "didnt", "nope"}
NEG_WINDOW = 4          # palavras antes do termo


@dataclass
class TermHit:
    category: str
    severity: str
    term: str            # trecho encontrado
    sentence: str
    negated: bool


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def _compile(cfg: dict):
    out = []
    for cat, spec in cfg["audio"]["critical_terms"].items():
        for pat in spec["patterns"]:
            # raiz de palavra: "vomit" casa vomit/vomiting/vomited; "faint" casa faint/fainted
            out.append((cat, spec["severity"], re.compile(r"\b" + re.escape(pat.lower()) + r"\w*", re.I)))
    return out


def _is_negated(sentence: str, start: int) -> bool:
    before = re.findall(r"[a-z']+", sentence[:start].lower())[-NEG_WINDOW:]
    return any(w in NEGATORS for w in before)


def find_critical_terms(text: str, cfg: dict | None = None) -> list[TermHit]:
    cfg = cfg or load_config()
    pats = _compile(cfg)
    hits: list[TermHit] = []
    for sent in split_sentences(text):
        seen: set[tuple[str, int]] = set()
        for cat, sev, rx in pats:
            for m in rx.finditer(sent):
                if (cat, m.start()) in seen:
                    continue
                seen.add((cat, m.start()))
                hits.append(TermHit(cat, sev, m.group(0), sent, _is_negated(sent, m.start())))
    return hits


# ---------------------------------------------------------------- sentimento
@dataclass
class SentimentResult:
    label: str           # POSITIVE | NEGATIVE
    score: float


class SentimentAnalyzer:
    name = "base"

    def analyze(self, sentences: list[str]) -> list[SentimentResult]:
        raise NotImplementedError


class TransformersSentiment(SentimentAnalyzer):
    """pipeline('sentiment-analysis'), como na aula 06. O modelo é baixado na 1ª execução."""
    name = "local"

    def __init__(self, cfg: dict | None = None, pipe=None):
        self.cfg = cfg or load_config()
        self._pipe = pipe

    def _load(self):
        if self._pipe is None:
            try:
                from transformers import pipeline
            except ImportError as e:
                raise RuntimeError("transformers não instalado (requirements-audio.txt)") from e
            model = self.cfg["audio"]["sentiment"].get("model")
            self._pipe = pipeline("sentiment-analysis", **({"model": model} if model else {}))
        return self._pipe

    def analyze(self, sentences):
        if not sentences:
            return []
        res = self._load()(sentences, truncation=True, max_length=512)
        return [SentimentResult(r["label"].upper(), float(r["score"])) for r in res]


class LexiconSentiment(SentimentAnalyzer):
    """Léxico mínimo, determinístico e sem download: só para desenvolvimento e testes."""
    name = "mock"
    NEG = {"pain", "painful", "bad", "terrible", "awful", "worse", "worried", "scared", "sick", "hurt",
           "hurts", "difficult", "struggling", "exhausted", "unbearable", "afraid", "horrible"}
    POS = {"good", "better", "fine", "great", "well", "improving", "relieved", "happy"}

    def analyze(self, sentences):
        out = []
        for s in sentences:
            w = set(re.findall(r"[a-z']+", s.lower()))
            n, p = len(w & self.NEG), len(w & self.POS)
            out.append(SentimentResult("NEGATIVE" if n > p else "POSITIVE", 0.95 if n != p else 0.5))
        return out


def get_sentiment_analyzer(name: str | None = None, cfg: dict | None = None) -> SentimentAnalyzer:
    name = (name or os.getenv("TEXT_ANALYZER") or "local").strip().lower()
    if name == "local":
        return TransformersSentiment(cfg)
    if name == "mock":
        return LexiconSentiment()
    if name == "azure":
        if not use_azure():
            raise RuntimeError("Azure desligado: defina USE_AZURE=true no .env para usar este provedor.")
        raise NotImplementedError("Azure Text Analytics será implementado na Task 10.")
    raise ValueError(f"Analisador desconhecido: {name!r}. Opções: local, mock, azure")


# ---------------------------------------------------------------- resultado
@dataclass
class TextAnalysis:
    terms: list[TermHit] = field(default_factory=list)
    n_sentences: int = 0
    negative_ratio: float | None = None
    most_negative: list[tuple[str, float]] = field(default_factory=list)
    analyzer: str = ""


def analyze_text(text: str, analyzer: SentimentAnalyzer | None = None, cfg: dict | None = None) -> TextAnalysis:
    cfg = cfg or load_config()
    analyzer = analyzer or get_sentiment_analyzer(cfg=cfg)
    sc = cfg["audio"]["sentiment"]
    sents = split_sentences(text)
    res = analyzer.analyze(sents)
    neg = [(s, r.score) for s, r in zip(sents, res) if r.label == "NEGATIVE" and r.score >= sc["negative_score_min"]]
    return TextAnalysis(
        terms=find_critical_terms(text, cfg), n_sentences=len(sents),
        negative_ratio=len(neg) / len(sents) if sents else None,
        most_negative=sorted(neg, key=lambda x: -x[1])[:3], analyzer=analyzer.name)


def evaluate_text(a: TextAnalysis, cfg: dict | None = None) -> list[Finding]:
    cfg = cfg or load_config()
    out: list[Finding] = []
    by_cat: dict[str, list[TermHit]] = {}
    for h in a.terms:
        if not h.negated:
            by_cat.setdefault(h.category, []).append(h)
    for cat, hs in by_cat.items():
        sev = hs[0].severity
        quote = hs[0].sentence if len(hs[0].sentence) <= 160 else hs[0].sentence[:157] + "..."
        out.append(Finding(f"termo:{cat}", sev,
                           f"Termo clínico '{hs[0].term}' ({cat}, {len(hs)}x): \"{quote}\"",
                           float(len(hs)), 1.0))
    sc = cfg["audio"]["sentiment"]
    if (a.negative_ratio is not None and a.n_sentences >= sc["min_segments"]
            and a.negative_ratio > sc["max_negative_ratio"]):
        out.append(Finding("tom_negativo", "low",
                           f"{100 * a.negative_ratio:.0f}% das frases com tom negativo",
                           a.negative_ratio, sc["max_negative_ratio"]))
    out.sort(key=lambda f: -SEVERITY_ORDER[f.severity])
    return out
