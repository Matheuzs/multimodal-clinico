"""Termos críticos e sentimento nas falas do paciente (PriMock57).

  python -m scripts.eval_text [--source human|whisper] [--analyzer local|mock] [--n 5]
`human` usa a transcrição humana (TextGrid); `whisper` usa outputs/audio/local_whisper_<modelo>/ (rode eval_asr antes).
"""
import argparse
import json

from src.audio.text_analysis import analyze_text, evaluate_text, get_sentiment_analyzer
from src.audio.textgrid import reference_text
from src.config import ROOT, load_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="human", choices=["human", "whisper"])
    ap.add_argument("--analyzer", default=None)
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    cfg = load_config()
    an = get_sentiment_analyzer(a.analyzer, cfg)
    base = ROOT / "data/raw/primock57"
    for i in range(1, a.n + 1):
        stem = f"day1_consultation{i:02d}_patient"
        if a.source == "human":
            text = reference_text(base / "transcripts" / f"{stem}.TextGrid")
        else:
            text = json.load(open(ROOT / f"outputs/audio/local_whisper_{cfg['audio']['whisper_model']}/{stem}.json"))["text"]
        res = analyze_text(text, an, cfg)
        print(f"\n== consulta {i:02d} ({a.source}, sentimento: {res.analyzer}): "
              f"{res.n_sentences} frases, {100 * (res.negative_ratio or 0):.0f}% negativas")
        for f in evaluate_text(res, cfg):
            print(f"  [{f.severity:6s}] {f.rule}: {f.message[:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
