import math

import pytest

import copy

from src.config import load_config
from src.vision.rules import analyze_repetitions, compute_baseline, evaluate_rules, segment_repetitions


def _cfg(**over):
    cfg = copy.deepcopy(load_config())
    cfg["vision"]["exercise"].update(over)
    return cfg


def _pose(peaks, rest=14.0, elbow=175.0, lead=10, gap=8, tail=10, fps=30.0, stride=3, other=25.0, n=20):
    """Série sintética: repouso -> sobe até `peak` -> volta, repetido."""
    shoulder = [rest] * lead
    for pk in peaks:
        up = [rest + (pk - rest) * math.sin(math.pi * i / n) for i in range(n + 1)]
        shoulder += up + [rest] * gap
    shoulder += [rest] * tail
    frames = [{"frame": i * stride, "t": i * stride / fps, "detected": True,
               "angles": {"right_shoulder": s, "right_elbow": elbow,
                          "left_shoulder": other, "left_elbow": 165.0}}
              for i, s in enumerate(shoulder)]
    return {"fps": fps, "frames": frames}


def test_segments_three_reps():
    assert len(segment_repetitions(_pose([135, 135, 135]))) == 3


def test_ignores_small_noise():
    assert segment_repetitions(_pose([50])) == []   # pico abaixo de rep_min_peak_deg


def test_truncated_start_is_dropped():
    pose = _pose([135, 135])
    for f in pose["frames"][:6]:                    # vídeo começa com braço já levantado
        f["angles"]["right_shoulder"] = 90.0
    assert len(segment_repetitions(pose)) == 2


def test_good_reps_not_flagged():
    reps = analyze_repetitions(_pose([138, 140]))
    assert len(reps) == 2 and not any(r.flagged for r in reps)


def test_absolute_amplitude_off_by_default():
    assert not analyze_repetitions(_pose([100, 100]))[0].flagged   # sem baseline (<3 reps) e regra desligada


@pytest.mark.parametrize("peak,sev", [(130, "low"), (120, "medium"), (100, "high")])
def test_absolute_amplitude_severity(peak, sev):
    rep = analyze_repetitions(_pose([peak]), _cfg(min_peak_shoulder_deg=133))[0]
    f = [x for x in rep.findings if x.rule == "amplitude"][0]
    assert f.severity == sev and rep.severity == sev


def test_bent_elbow_and_compensation():
    rep = analyze_repetitions(_pose([138], elbow=110.0))[0]
    assert {f.rule for f in rep.findings} == {"cotovelo_dobrado"}
    pose = _pose([138])
    for i, f in enumerate(pose["frames"]):
        f["angles"]["left_shoulder"] = 20 + (i % 2) * 30   # ombro oposto oscila
    assert "compensacao" in {f.rule for f in analyze_repetitions(pose)[0].findings}


def test_low_quality_flag():
    f = evaluate_rules({"peak_shoulder": 138, "min_elbow": 160, "other_shoulder_range": 5,
                        "detected_ratio": 0.5})
    assert [x.rule for x in f] == ["qualidade"]


def test_missing_angles_are_interpolated():
    pose = _pose([138])
    for f in pose["frames"][14:17]:
        f["angles"]["right_shoulder"] = None
    assert len(segment_repetitions(pose)) == 1


def test_baseline_needs_three_reps():
    assert compute_baseline([{"peak_shoulder": 1, "speed_up": 1, "min_elbow": 1}] * 2) is None


def test_relative_amplitude_both_directions():
    lower = analyze_repetitions(_pose([138, 138, 138, 118]))
    higher = analyze_repetitions(_pose([118, 118, 118, 150]))
    assert [r.flagged for r in lower] == [False, False, False, True]
    assert [r.flagged for r in higher] == [False, False, False, True]
    assert "menor" in lower[3].findings[0].message and "maior" in higher[3].findings[0].message


def test_relative_speed_both_directions():
    base = [138, 138, 138]
    fast = analyze_repetitions(_pose(base + [138], n=20))   # mesma velocidade: ok
    assert not any(r.flagged for r in fast)
    # repetição mais lenta/rápida: simula alongando o movimento (n maior = mais lento)
    slow = _pose(base, n=20)
    slow_rep = _pose([138], n=40)["frames"]
    off = slow["frames"][-1]["frame"] + 3
    for i, f in enumerate(slow_rep):
        f["frame"] = off + 3 * i
    slow["frames"] += slow_rep
    reps = analyze_repetitions(slow)
    assert "desvio_velocidade" in {f.rule for f in reps[-1].findings}


def test_relative_elbow_deviation():
    pose = _pose([138, 138, 138, 138], elbow=170.0)
    for f in pose["frames"][-25:]:
        f["angles"]["right_elbow"] = 150.0
    rep = analyze_repetitions(pose)[-1]
    assert "desvio_cotovelo" in {f.rule for f in rep.findings}
