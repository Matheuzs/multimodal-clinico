import pytest

from src.config import load_config
from src.findings import Finding
from src.fusion.fusion import fuse
from src.vision.rules import Repetition


def rep(i, *findings):
    return Repetition(i, 0, 1, 0, {}, list(findings))


def F(rule, sev):
    return Finding(rule, sev, f"msg {rule}", 1.0, 1.0)


@pytest.fixture
def cfg():
    return load_config()


def test_all_normal(cfg):
    r = fuse([rep(i) for i in range(5)], [], cfg)
    assert r.level == "NORMAL" and r.score == 0 and not r.is_alert


def test_same_rule_counts_once(cfg):
    reps = [rep(i, F("desvio_amplitude", "medium")) for i in range(5)]
    r = fuse(reps, None, cfg)
    assert r.vision_score == 2 and r.level == "ATENÇÃO"
    assert r.evidences[0].detail == "repetições 0, 1, 2, 3, 4 de 5"


def test_isolated_vision_noise_ignored(cfg):
    reps = [rep(0, F("desvio_velocidade", "medium"))] + [rep(i) for i in range(1, 10)]   # 10% < 20%
    assert fuse(reps, None, cfg).level == "NORMAL"


def test_convergence_raises_to_alert(cfg):
    reps = [rep(i, F("desvio_amplitude", "medium")) for i in range(3)] + [rep(3), rep(4)]
    only_v = fuse(reps, None, cfg)
    only_a = fuse(None, [F("pausas_longas", "medium")], cfg)
    both = fuse(reps, [F("pausas_longas", "medium")], cfg)
    assert only_v.level == only_a.level == "ATENÇÃO"
    assert both.convergence and both.score == 5 and both.level == "ALERTA" and both.is_alert


def test_high_override(cfg):
    r = fuse(None, [F("termo:chest_pain", "high")], cfg)
    assert r.score == 3 and r.level == "ALERTA"
    cfg["fusion"]["high_override"] = False
    assert fuse(None, [F("termo:chest_pain", "high")], cfg).level == "ATENÇÃO"


def test_low_only_stays_normal(cfg):
    assert fuse(None, [F("tom_negativo", "low")], cfg).level == "NORMAL"


def test_single_modality_lists_modalities(cfg):
    assert fuse(None, [], cfg).modalities == ["audio"]
    assert fuse([], None, cfg).modalities == ["video"]
    assert fuse([], [], cfg).modalities == ["video", "audio"]


def test_to_dict_serializable(cfg):
    import json
    r = fuse([rep(0, F("x", "high"))], [F("y", "medium")], cfg)
    d = json.loads(json.dumps(r.to_dict()))
    assert d["level"] == "ALERTA" and d["evidences"][0]["severity"] == "high"
    assert "Requer avaliação profissional" in d["message"]


def test_many_mild_rules_single_modality_is_only_attention(cfg):
    fs = [F(f"termo:{c}", "medium") for c in "abcd"]       # score 8, mas uma só fonte
    r = fuse(None, fs, cfg)
    assert r.score == 8 and r.level == "ATENÇÃO"
    cfg["fusion"]["alert_requires_convergence"] = False
    assert fuse(None, fs, cfg).level == "ALERTA"
