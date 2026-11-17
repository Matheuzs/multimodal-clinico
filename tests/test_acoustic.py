import numpy as np
import pytest

from src.audio.acoustic import AcousticFeatures, evaluate_acoustic, extract_features
from src.audio.vad import frame_f1, overlap, speech_intervals
from src.config import ROOT

SR = 16000


def _speech(spec, sr=SR):
    """spec: lista de (duração_s, amplitude); amplitude 0 = silêncio. Ruído branco como 'fala'."""
    rng = np.random.default_rng(0)
    return np.concatenate([a * rng.standard_normal(int(d * sr)).astype("float32") + 1e-5 * rng.standard_normal(int(d * sr)).astype("float32")
                           for d, a in spec]).astype("float32")


def _write(path, y, sr=SR):
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32000).astype("<i2").tobytes())
    return path


def test_vad_finds_two_bursts():
    y = _speech([(1, 0), (2, 0.1), (1, 0), (2, 0.1), (1, 0)])
    iv = speech_intervals(y, SR)
    assert len(iv) == 2
    assert iv[0][0] == pytest.approx(1.0, abs=0.1) and iv[0][1] == pytest.approx(3.0, abs=0.1)


def test_vad_merges_short_gaps():
    y = _speech([(1, 0.1), (0.1, 0), (1, 0.1)])
    assert len(speech_intervals(y, SR)) == 1


def test_overlap_and_f1():
    assert overlap((1, 3), [(0, 2), (2.5, 4)]) == pytest.approx(1.5)
    f = frame_f1([(0, 2)], [(0, 2)], 3)
    assert f["f1"] == pytest.approx(1.0)


def test_pauses_counted_and_long(tmp_path):
    y = _speech([(2, 0.1), (0.5, 0), (2, 0.1), (2.0, 0), (2, 0.1)])
    f = extract_features(_write(tmp_path / "a.wav", y), n_words=60)
    assert f.n_pauses == 2 and f.n_long_pauses == 1
    assert f.max_pause_s == pytest.approx(2.0, abs=0.15)
    assert f.speech_s == pytest.approx(6.0, abs=0.3)
    assert f.speech_rate_wps == pytest.approx(10.0, abs=0.6)


def test_turn_gap_excluded_without_other_channel(tmp_path):
    y = _speech([(2, 0.1), (8, 0), (2, 0.1)])          # 8 s > turn_gap_s (5): troca de turno
    assert extract_features(_write(tmp_path / "a.wav", y)).n_pauses == 0


def test_other_channel_masks_turns(tmp_path):
    pat = _speech([(2, 0.1), (4, 0), (2, 0.1), (3, 0), (2, 0.1)])
    doc = _speech([(2, 0), (4, 0.1), (2, 0), (3, 0), (2, 0)])   # médico fala só no 1º intervalo
    f = extract_features(_write(tmp_path / "p.wav", pat), other_wav_path=_write(tmp_path / "d.wav", doc))
    assert f.n_pauses == 1 and f.max_pause_s == pytest.approx(3.0, abs=0.2)   # o de 4 s era turno


def test_energy_trend_negative(tmp_path):
    y = _speech([(3, 0.2), (3, 0.1), (3, 0.05)])
    f = extract_features(_write(tmp_path / "a.wav", y))
    assert f.energy_trend_db < -6


def _feat(**kw):
    base = dict(duration_s=300, speech_s=150, active_s=180, n_pauses=10, n_long_pauses=2, pause_s=30,
                pause_ratio=0.16, max_pause_s=2.0, long_pauses_per_min=0.7, rms_db_mean=-30,
                energy_trend_db=-1.0, n_words=450, speech_rate_wps=3.0)
    base.update(kw)
    return AcousticFeatures(**base)


def test_typical_speech_not_flagged():
    assert evaluate_acoustic(_feat()) == []


def test_flags_and_severity():
    r = {f.rule: f for f in evaluate_acoustic(_feat(speech_rate_wps=0.7, long_pauses_per_min=7.0,
                                                    pause_ratio=0.5, energy_trend_db=-12))}
    assert set(r) == {"fala_lenta", "pausas_longas", "muita_pausa", "queda_energia"}
    assert r["fala_lenta"].severity == "high" and r["pausas_longas"].severity == "high"
    assert r["queda_energia"].severity == "medium"      # 12 dB / 6 dB = 2x
    mild = {f.rule: f for f in evaluate_acoustic(_feat(speech_rate_wps=1.5))}
    assert mild["fala_lenta"].severity == "low"


def test_missing_optional_features_do_not_flag():
    assert evaluate_acoustic(_feat(speech_rate_wps=None, energy_trend_db=None)) == []


def test_real_primock_patient_is_plausible():
    b = ROOT / "data/raw/primock57/audio"
    if not (b / "day1_consultation01_patient.wav").exists():
        pytest.skip("PriMock57 não baixado")
    f = extract_features(b / "day1_consultation01_patient.wav", 438, b / "day1_consultation01_doctor.wav")
    assert 100 < f.speech_s < 250 and 2 < f.speech_rate_wps < 4
    assert evaluate_acoustic(f) == []          # ator saudável: sem alertas
