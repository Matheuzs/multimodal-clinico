"""Leitor mínimo de TextGrid (formato Praat) para as transcrições do PriMock57."""
from __future__ import annotations

import re
from pathlib import Path

_INTERVAL = re.compile(
    r'xmin\s*=\s*([\d.eE+-]+)\s*xmax\s*=\s*([\d.eE+-]+)\s*text\s*=\s*"((?:[^"]|"")*)"', re.S)
_TAGS = re.compile(r"</?[A-Z_]+/?>")          # <UNSURE>, </UNSURE>, <UNIN/>, <INAUDIBLE_SPEECH/>


def clean_text(text: str) -> str:
    """Remove as marcações do transcritor; o conteúdo de <UNSURE>...</UNSURE> é mantido."""
    return re.sub(r"\s+", " ", _TAGS.sub(" ", text.replace('""', '"'))).strip()


def read_textgrid(path: str | Path) -> list[tuple[float, float, str]]:
    """Retorna [(início_s, fim_s, texto_limpo)] só dos intervalos com fala."""
    raw = Path(path).read_text(encoding="utf-8")
    out = []
    for a, b, t in _INTERVAL.findall(raw):
        t = clean_text(t)
        if t:
            out.append((float(a), float(b), t))
    return out


def reference_text(path: str | Path) -> str:
    return " ".join(t for _, _, t in read_textgrid(path))
