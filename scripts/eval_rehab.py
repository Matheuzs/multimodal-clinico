"""Valida segmentação e regras contra o Segmentation.csv do REHAB24-6.

  python scripts/eval_rehab.py PM_001 [PM_000 ...]   # usa data/raw/rehab24/videos/<id>-Camera17-30fps.mp4
"""
import csv
import sys
from collections import Counter

from src.config import ROOT, load_config
from src.vision.pose_analyzer import analyze_video
from src.vision.rules import analyze_repetitions


def main(ids: list[str]) -> int:
    cfg = load_config()
    labels = list(csv.DictReader(open(ROOT / "data/raw/rehab24/Segmentation.csv"), delimiter=";"))
    conf = Counter()
    for vid in ids:
        path = ROOT / f"data/raw/rehab24/videos/{vid}-Camera17-30fps.mp4"
        pose = analyze_video(path, cfg).to_json()
        reps = analyze_repetitions(pose, cfg)
        truth = [r for r in labels if r["video_id"] == vid]
        print(f"\n== {vid}: {len(reps)} repetições detectadas, {len(truth)} no rótulo")
        for rep in reps:
            m = [t for t in truth if int(t["first_frame"]) <= rep.peak_frame <= int(t["last_frame"])]
            real = ("correta" if m[0]["correctness"] == "1" else "ERRADA") if m else "sem rótulo"
            pred = f"sinalizada ({rep.severity}: {', '.join(f.rule.replace('desvio_', '') for f in rep.findings)})" if rep.flagged else "ok"
            print(f"  rep {rep.index:2d} frames {rep.start_frame}-{rep.end_frame} "
                  f"pico={rep.features['peak_shoulder']:.0f}° cot={rep.features['min_elbow']:.0f}° vel={rep.features['speed_up']:.0f}°/s | rótulo: {real:10s} | regras: {pred}")
            if m:
                conf[(m[0]["correctness"], rep.flagged)] += 1
    tp, fn = conf[("0", True)], conf[("0", False)]
    tn, fp = conf[("1", False)], conf[("1", True)]
    print(f"\nErradas sinalizadas: {tp}/{tp + fn} | Corretas não sinalizadas: {tn}/{tn + fp}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["PM_001"]))
