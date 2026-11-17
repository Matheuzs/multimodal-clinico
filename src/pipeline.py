"""Fluxo completo: vídeo + áudio -> achados -> fusão -> relatório.

  python -m src.pipeline --video V.mp4 [--start N --end M] --audio paciente.wav [--other-audio medico.wav]
         [--transcriber local_whisper|google|mock] [--analyzer local|mock] [--out outputs/report] [--annotate]

Qualquer modalidade pode ser omitida. Gera report.md, report.html e report.json em --out.
Os componentes pesados (pose, ASR, sentimento) podem ser injetados, o que permite testar sem modelos.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.config import ROOT, load_config
from src.findings import Finding
from src.fusion.fusion import fuse
from src.ingestion import ingest


def _f(f: Finding) -> dict:
    return {"rule": f.rule, "severity": f.severity, "message": f.message,
            "value": round(float(f.value), 3), "threshold": round(float(f.threshold), 3)}


def run_vision(video, cfg, start=0, end=None, annotate_to=None, pose_fn=None, progress=False):
    """Retorna (dict para o relatório, lista de Repetition)."""
    from src.vision.rules import analyze_repetitions
    info = ingest(video, cfg)
    if pose_fn is None:
        from src.vision.pose_analyzer import analyze_video
        pose = analyze_video(video, cfg, start=start, end=end, annotate_to=annotate_to,
                             progress=progress).to_json()
    else:
        pose = pose_fn(video)
    reps = analyze_repetitions(pose, cfg)
    fps = pose["fps"]
    data = {
        "file": Path(video).name, "duration_s": info.duration_s, "fps": fps,
        "frames_analyzed": pose.get("n_frames", len(pose["frames"])),
        "frames_with_pose": pose.get("n_detected", sum(f["detected"] for f in pose["frames"])),
        "annotated_video": str(annotate_to) if annotate_to else None,
        "repetitions": [{
            "index": r.index, "start_s": round(r.start_frame / fps, 2), "end_s": round(r.end_frame / fps, 2),
            "features": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.features.items()},
            "severity": r.severity, "findings": [_f(x) for x in r.findings]} for r in reps],
    }
    return data, reps


def run_audio(audio, cfg, other_audio=None, transcriber=None, sentiment=None, transcriber_name=None, analyzer_name=None):
    """Retorna (dict para o relatório, lista de Finding)."""
    from src.audio.acoustic import evaluate_acoustic, extract_features
    from src.audio.metrics import normalize
    from src.audio.text_analysis import analyze_text, evaluate_text, get_sentiment_analyzer
    from src.audio.transcriber import get_transcriber
    info = ingest(audio, cfg)
    tr = transcriber or get_transcriber(transcriber_name, cfg)
    t = tr.transcribe(audio)
    n_words = len(normalize(t.text))
    feats = extract_features(audio, n_words, other_audio, cfg)
    ac = evaluate_acoustic(feats, cfg)
    an = analyze_text(t.text, sentiment or get_sentiment_analyzer(analyzer_name, cfg), cfg)
    tx = evaluate_text(an, cfg)
    data = {
        "file": Path(audio).name, "duration_s": info.duration_s,
        "transcriber": t.provider, "sentiment_analyzer": an.analyzer, "n_words": n_words,
        "transcript": t.text,
        "acoustic": feats.to_dict(), "acoustic_findings": [_f(x) for x in ac],
        "terms": [{"category": h.category, "severity": h.severity, "term": h.term,
                   "sentence": h.sentence, "negated": h.negated} for h in an.terms],
        "negative_ratio": an.negative_ratio, "n_sentences": an.n_sentences,
        "most_negative": [{"sentence": s, "score": round(sc, 3)} for s, sc in an.most_negative],
        "text_findings": [_f(x) for x in tx],
    }
    return data, ac + tx


def run(video=None, audio=None, other_audio=None, out_dir=None, cfg=None, start=0, end=None, annotate=False,
        pose_fn=None, transcriber=None, sentiment=None, transcriber_name=None, analyzer_name=None, progress=False):
    if video is None and audio is None:
        raise ValueError("Informe ao menos um de: vídeo, áudio")
    from src.report.report import render_html, render_markdown
    cfg = cfg or load_config()
    out = Path(out_dir) if out_dir else ROOT / "outputs" / "report"
    out.mkdir(parents=True, exist_ok=True)
    vdata = adata = reps = afind = None
    if video is not None:
        ann = out / f"{Path(video).stem}_annotated.mp4" if annotate else None
        vdata, reps = run_vision(video, cfg, start, end, ann, pose_fn, progress)
    if audio is not None:
        adata, afind = run_audio(audio, cfg, other_audio, transcriber, sentiment, transcriber_name, analyzer_name)
    fusion = fuse(reps, afind, cfg)
    report = {"generated_at": datetime.now().isoformat(timespec="seconds"),
              "fusion": fusion.to_dict(), "vision": vdata, "audio": adata}
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(render_markdown(report), encoding="utf-8")
    (out / "report.html").write_text(render_html(report), encoding="utf-8")
    return report, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--audio")
    ap.add_argument("--other-audio", help="canal do interlocutor (médico), usado para descontar trocas de turno")
    ap.add_argument("--transcriber", default=None)
    ap.add_argument("--analyzer", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not (a.video or a.audio):
        ap.error("informe --video e/ou --audio")
    report, out = run(a.video, a.audio, a.other_audio, a.out, start=a.start, end=a.end, annotate=a.annotate,
                      transcriber_name=a.transcriber, analyzer_name=a.analyzer, progress=True)
    fz = report["fusion"]
    print(f"\nNível: {fz['level']}  (score {fz['score']:g})\n{fz['message']}\nRelatório: {out}/report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
