import cv2
import numpy as np
import pytest

from src.ingestion import IngestionError, ingest


def test_video_valid(make_video):
    info = ingest(make_video(seconds=2, fps=10))
    assert info.kind == "video" and info.frames == 20
    assert info.width == 64 and info.height == 48
    assert info.duration_s == pytest.approx(2.0, abs=0.2)


def test_video_too_long(make_video):
    cfg = {"ingestion": {"video_exts": [".mp4"], "image_exts": [], "audio_exts": [],
                         "max_video_seconds": 1, "target_audio_sr": 16000}}
    with pytest.raises(IngestionError, match="excede"):
        ingest(make_video(seconds=3), cfg)


def test_video_corrupted(tmp_path):
    p = tmp_path / "bad.mp4"
    p.write_bytes(b"not a video at all")
    with pytest.raises(IngestionError):
        ingest(p)


def test_image_valid_and_invalid(tmp_path):
    ok = tmp_path / "i.png"
    cv2.imwrite(str(ok), np.zeros((30, 40, 3), np.uint8))
    info = ingest(ok)
    assert (info.width, info.height) == (40, 30)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"xxx")
    with pytest.raises(IngestionError):
        ingest(bad)


def test_audio_valid(make_wav):
    info = ingest(make_wav(seconds=1.5))
    assert info.kind == "audio" and info.sample_rate == 16000
    assert info.duration_s == pytest.approx(1.5, abs=0.01) and not info.warnings


def test_audio_warns_on_rate_and_channels(make_wav):
    info = ingest(make_wav(sr=44100, channels=2))
    assert len(info.warnings) == 2


def test_audio_invalid(tmp_path):
    p = tmp_path / "bad.wav"
    p.write_bytes(b"RIFFxxxx")
    with pytest.raises(IngestionError):
        ingest(p)


def test_missing_empty_unsupported(tmp_path):
    with pytest.raises(IngestionError, match="não encontrado"):
        ingest(tmp_path / "nope.mp4")
    e = tmp_path / "e.wav"
    e.write_bytes(b"")
    with pytest.raises(IngestionError, match="vazio"):
        ingest(e)
    t = tmp_path / "x.txt"
    t.write_text("hi")
    with pytest.raises(IngestionError, match="não suportado"):
        ingest(t)


def test_real_primock_audio_if_present():
    from src.config import ROOT
    f = ROOT / "data/raw/primock57/audio/day1_consultation01_patient.wav"
    if not f.exists():
        pytest.skip("PriMock57 não baixado")
    info = ingest(f)
    assert info.sample_rate == 16000 and info.channels == 1 and info.duration_s > 60
