"""Normalização de texto e WER (taxa de erro de palavras)."""
from __future__ import annotations

import re

# Interjeições que o Whisper costuma omitir e as transcrições humanas mantêm: removidas dos dois lados.
FILLERS = {"um", "uh", "umm", "uhm", "erm", "er", "mm", "mhm", "hmm", "ah", "eh"}


def normalize(text: str) -> list[str]:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return [w.strip("'") for w in text.split() if w.strip("'") and w.strip("'") not in FILLERS]


def wer(reference: str, hypothesis: str) -> dict:
    """WER = (S + D + I) / N, por distância de edição entre palavras."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    n, m = len(ref), len(hyp)
    # d[i][j] = (custo, S, D, I)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur.append(prev[j - 1])
                continue
            s, d, ins = prev[j - 1], prev[j], cur[j - 1]
            best = min((s[0], "s"), (d[0], "d"), (ins[0], "i"))
            if best[1] == "s":
                cur.append((s[0] + 1, s[1] + 1, s[2], s[3]))
            elif best[1] == "d":
                cur.append((d[0] + 1, d[1], d[2] + 1, d[3]))
            else:
                cur.append((ins[0] + 1, ins[1], ins[2], ins[3] + 1))
        prev = cur
    cost, sub, dele, ins = prev[m]
    return {"wer": cost / n if n else (0.0 if m == 0 else 1.0), "ref_words": n,
            "substitutions": sub, "deletions": dele, "insertions": ins}
