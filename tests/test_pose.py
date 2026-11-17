import math

import numpy as np
import pytest

from src.config import ROOT
from src.vision.pose_analyzer import LM, angle_deg, compute_angles


def test_angle_right_angle():
    assert angle_deg((1, 0), (0, 0), (0, 1)) == pytest.approx(90)


def test_angle_straight_and_folded():
    assert angle_deg((-1, 0), (0, 0), (1, 0)) == pytest.approx(180)
    assert angle_deg((1, 0), (0, 0), (2, 0)) == pytest.approx(0)


def test_angle_degenerate_is_nan():
    assert math.isnan(angle_deg((0, 0), (0, 0), (1, 1)))


def _landmarks(vis=1.0):
    lm = np.zeros((33, 4), np.float32)
    lm[:, 3] = vis
    lm[LM["right_shoulder"]] = (0.5, 0.2, 0, vis)
    lm[LM["right_elbow"]] = (0.5, 0.4, 0, vis)
    lm[LM["right_wrist"]] = (0.7, 0.4, 0, vis)   # cotovelo dobrado em 90°
    lm[LM["right_hip"]] = (0.5, 0.6, 0, vis)
    return lm


def test_compute_angles_elbow_90():
    ang = compute_angles(_landmarks(), 1000, 1000, 0.5)
    assert ang["right_elbow"] == pytest.approx(90, abs=0.1)
    assert ang["right_shoulder"] == pytest.approx(0, abs=0.1)  # braço pendurado: quadril e cotovelo na mesma direção


def test_compute_angles_low_visibility_is_none():
    ang = compute_angles(_landmarks(vis=0.1), 1000, 1000, 0.5)
    assert all(v is None for v in ang.values())


def test_aspect_ratio_matters():
    # Ângulo deve ser calculado em pixels: em 16:9 difere do calculado em coordenadas normalizadas.
    lm = _landmarks()
    lm[LM["right_wrist"]] = (0.6, 0.5, 0, 1.0)            # diagonal
    norm = compute_angles(lm, 1000, 1000, 0.5)["right_elbow"]   # imagem quadrada = normalizado
    wide = compute_angles(lm, 1920, 1080, 0.5)["right_elbow"]
    assert norm == pytest.approx(135, abs=0.1)
    assert wide == pytest.approx(119.4, abs=0.5)


def test_analyze_real_video_if_present():
    pytest.importorskip("mediapipe")
    from src.vision.pose_analyzer import analyze_video
    v = ROOT / "data/raw/rehab24/videos/PM_001-Camera17-30fps.mp4"
    if not v.exists():
        pytest.skip("vídeo do REHAB24-6 não baixado")
    res = analyze_video(v, start=300, end=330)
    assert len(res.frame_idx) == 11 and res.n_detected >= 9   # frames 300..330, passo 3
    first = next(a for a in res.angles if a)
    assert set(first) == {"right_elbow", "left_elbow", "right_shoulder", "left_shoulder"}


def test_analyze_blank_video_no_crash(make_video):
    pytest.importorskip("mediapipe")
    from src.vision.pose_analyzer import analyze_video
    res = analyze_video(make_video(seconds=1, fps=10, size=(160, 120)))
    assert res.n_detected == 0 and len(res.frame_idx) > 0
