"""Análise de pose em vídeo com MediaPipe Pose (mp.solutions.pose, como na aula 03).

Lê o vídeo frame a frame (1 a cada `frame_stride`), extrai os 33 landmarks e calcula
ângulos de cotovelo e ombro. Saídas: série temporal (JSON), landmarks (NPZ) e,
opcionalmente, vídeo anotado.

Uso:
  python -m src.vision.pose_analyzer VIDEO [--start 180 --end 377] [--annotate]
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.config import ROOT, load_config
from src.ingestion import ingest

# Índices dos landmarks do MediaPipe Pose (nomes referem-se ao lado da PESSOA).
LM = {
    "left_shoulder": 11, "right_shoulder": 12,
    "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16,
    "left_hip": 23, "right_hip": 24,
}
# ângulo -> (ponto A, vértice B, ponto C)
ANGLE_DEFS = {
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_shoulder": ("right_hip", "right_shoulder", "right_elbow"),
    "left_shoulder": ("left_hip", "left_shoulder", "left_elbow"),
}


def angle_deg(a, b, c) -> float:
    """Ângulo em graus no vértice b, formado por a-b-c (2D)."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos))


def compute_angles(landmarks: np.ndarray, width: int, height: int,
                   visibility_min: float) -> dict[str, float | None]:
    """landmarks: (33, 4) = x, y (normalizados), z, visibility. Usa pixels p/ não distorcer ângulos."""
    out: dict[str, float | None] = {}
    for name, (a, b, c) in ANGLE_DEFS.items():
        pts = [landmarks[LM[k]] for k in (a, b, c)]
        if min(p[3] for p in pts) < visibility_min:
            out[name] = None
            continue
        px = [(p[0] * width, p[1] * height) for p in pts]
        ang = angle_deg(*px)
        out[name] = None if math.isnan(ang) else round(ang, 2)
    return out


@dataclass
class PoseResult:
    video: str
    fps: float
    stride: int
    width: int
    height: int
    frame_idx: list[int] = field(default_factory=list)
    landmarks: list = field(default_factory=list)   # (33,4) ou None por frame
    angles: list = field(default_factory=list)      # dict por frame (vazio se sem detecção)

    @property
    def n_detected(self) -> int:
        return sum(lm is not None for lm in self.landmarks)

    def to_json(self) -> dict:
        return {
            "video": self.video, "fps": self.fps, "stride": self.stride,
            "width": self.width, "height": self.height,
            "n_frames": len(self.frame_idx), "n_detected": self.n_detected,
            "frames": [
                {"frame": f, "t": round(f / self.fps, 3),
                 "detected": lm is not None, "angles": ang}
                for f, lm, ang in zip(self.frame_idx, self.landmarks, self.angles)
            ],
        }

    def save(self, out_dir: Path) -> tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(self.video).stem
        jpath, npath = out_dir / f"{stem}_pose.json", out_dir / f"{stem}_landmarks.npz"
        jpath.write_text(json.dumps(self.to_json(), indent=1), encoding="utf-8")
        arr = np.stack([lm if lm is not None else np.full((33, 4), np.nan)
                        for lm in self.landmarks]) if self.landmarks else np.empty((0, 33, 4))
        np.savez_compressed(npath, frames=np.array(self.frame_idx), landmarks=arr)
        return jpath, npath


def analyze_video(video_path, cfg: dict | None = None, start: int = 0,
                  end: int | None = None, annotate_to: Path | None = None,
                  progress: bool = False) -> PoseResult:
    import mediapipe as mp  # import tardio: dependência pesada (requirements-vision.txt)

    cfg = cfg or load_config()
    vcfg = cfg["vision"]
    info = ingest(video_path, cfg)
    if info.kind != "video":
        raise ValueError(f"{video_path} não é um vídeo")
    stride = int(vcfg["frame_stride"])
    proc_w = int(vcfg.get("process_width", 640))
    vis_min = vcfg["thresholds"]["visibility_min"]
    end = info.frames if end is None else min(end + 1, info.frames)

    mp_pose, mp_draw = mp.solutions.pose, mp.solutions.drawing_utils
    res = PoseResult(Path(video_path).name, info.fps, stride, info.width, info.height)
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    writer = None
    todo = range(start, end)
    iterator = todo
    if progress:
        from tqdm import tqdm
        iterator = tqdm(todo, desc=res.video)

    with mp_pose.Pose(model_complexity=vcfg["pose_model_complexity"],
                      min_detection_confidence=vcfg["min_detection_confidence"]) as pose:
        for idx in iterator:
            ok, frame = cap.read()
            if not ok:
                break
            if (idx - start) % stride:
                continue
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (proc_w, int(h * proc_w / w))) if w > proc_w else frame
            result = pose.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            res.frame_idx.append(idx)
            if result.pose_landmarks:
                lm = np.array([[p.x, p.y, p.z, p.visibility]
                               for p in result.pose_landmarks.landmark], dtype=np.float32)
                res.landmarks.append(lm)
                res.angles.append(compute_angles(lm, w, h, vis_min))
            else:
                res.landmarks.append(None)
                res.angles.append({})

            if annotate_to is not None:
                aw = int(vcfg.get("annotate_width", 960))
                vis = cv2.resize(frame, (aw, int(h * aw / w)))
                if result.pose_landmarks:
                    mp_draw.draw_landmarks(vis, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    y = 28
                    for name, val in res.angles[-1].items():
                        if val is not None:
                            cv2.putText(vis, f"{name}: {val:.0f}", (10, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            y += 28
                cv2.putText(vis, f"frame {idx}", (10, vis.shape[0] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if writer is None:
                    annotate_to.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(annotate_to), cv2.VideoWriter_fourcc(*"mp4v"),
                                             max(info.fps / stride, 1), (vis.shape[1], vis.shape[0]))
                writer.write(vis)
    cap.release()
    if writer is not None:
        writer.release()
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "vision"))
    ap.add_argument("--annotate", action="store_true", help="gera vídeo anotado")
    a = ap.parse_args()
    out = Path(a.out)
    ann = out / f"{Path(a.video).stem}_annotated.mp4" if a.annotate else None
    res = analyze_video(a.video, start=a.start, end=a.end, annotate_to=ann, progress=True)
    j, n = res.save(out)
    print(f"{res.n_detected}/{len(res.frame_idx)} frames com pose -> {j}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
