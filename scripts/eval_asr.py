"""Mede o WER da transcrição das falas do PACIENTE no PriMock57.

  python -m scripts.eval_asr [--provider local_whisper|google|mock] [--model tiny|base|small|medium] [--n 3]
Salva transcrições em outputs/audio/<provider>[_<modelo>]/.
"""
import argparse
import json
import time

from src.audio.metrics import wer
from src.audio.textgrid import reference_text
from src.audio.transcriber import get_transcriber
from src.config import ROOT, load_config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    ap.add_argument("--n", type=int, default=3, help="nº de consultas")
    ap.add_argument("--model", default=None, help="modelo do Whisper (padrão: config.yaml)")
    a = ap.parse_args()
    cfg = load_config()
    if a.model:
        cfg["audio"]["whisper_model"] = a.model
    tr = get_transcriber(a.provider, cfg)
    base = ROOT / "data/raw/primock57"
    wavs = sorted((base / "audio").glob("*_patient.wav"))[: a.n]
    label = f"{tr.name}_{cfg['audio']['whisper_model']}" if tr.name == "local_whisper" else tr.name
    out = ROOT / "outputs/audio" / label
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for wav in wavs:
        t0 = time.time()
        hyp = tr.transcribe(wav)
        ref = reference_text(base / "transcripts" / f"{wav.stem}.TextGrid")
        m = wer(ref, hyp.text)
        (out / f"{wav.stem}.json").write_text(json.dumps(
            {"text": hyp.text, "segments": hyp.segments, **m}, indent=1, ensure_ascii=False), encoding="utf-8")
        rows.append(m)
        print(f"{wav.stem}: WER={100 * m['wer']:.1f}% ({m['ref_words']} palavras; "
              f"S={m['substitutions']} D={m['deletions']} I={m['insertions']}) em {time.time() - t0:.0f}s")
    tot = sum(r["ref_words"] for r in rows)
    err = sum(r["wer"] * r["ref_words"] for r in rows)
    print(f"\nWER médio ponderado ({label}): {100 * err / tot:.1f}% em {len(rows)} áudios")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
