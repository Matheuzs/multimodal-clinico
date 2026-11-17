import json

import pytest

from src.audio.text_analysis import LexiconSentiment
from src.audio.transcriber import Transcriber, Transcript
from src.pipeline import run
from tests.test_rules import _pose


class FakeTranscriber(Transcriber):
    name = "fake"

    def __init__(self, text):
        self.text = text

    def transcribe(self, wav):
        return Transcript(self.text, "fake", "en", [(0, 1, self.text)])


def _run(tmp_path, make_video, make_wav, peaks, text, **kw):
    pose_fn = (lambda v: _pose(peaks)) if peaks is not None else None
    return run(video=make_video("v.mp4") if peaks is not None else None,
               audio=make_wav("p.wav", 3.0) if text is not None else None,
               out_dir=tmp_path / "out", pose_fn=pose_fn,
               transcriber=FakeTranscriber(text) if text is not None else None,
               sentiment=LexiconSentiment(), **kw)


def test_normal_case(tmp_path, make_video, make_wav):
    rep, out = _run(tmp_path, make_video, make_wav, [138, 140, 139, 141], "I feel fine today, thank you.")
    assert rep["fusion"]["level"] == "NORMAL"
    for name in ("report.md", "report.json", "report.html"):
        assert (out / name).stat().st_size > 0
    assert "NORMAL" in (out / "report.md").read_text(encoding="utf-8")


def test_alert_from_critical_term_with_vision_deviation(tmp_path, make_video, make_wav):
    rep, out = _run(tmp_path, make_video, make_wav, [140, 140, 140, 90, 90, 90],
                    "I have chest pain when I raise my arm. I have chest pain every day.")
    fz = rep["fusion"]
    assert fz["level"] == "ALERTA" and fz["convergence"] is True
    assert {e["modality"] for e in fz["evidences"]} == {"video", "audio"}
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "chest_pain" in md and "Requer avaliação profissional" in md
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "class='badge ALERTA'" in html and "<table>" in html


def test_audio_only_and_video_only(tmp_path, make_video, make_wav):
    r1, _ = _run(tmp_path, make_video, make_wav, None, "I have chest pain.")
    assert r1["vision"] is None and r1["fusion"]["modalities"] == ["audio"] and r1["fusion"]["level"] == "ALERTA"
    r2, _ = _run(tmp_path / "b", make_video, make_wav, [140, 140, 140], None)
    assert r2["audio"] is None and r2["fusion"]["modalities"] == ["video"]


def test_json_roundtrip_and_negation(tmp_path, make_video, make_wav):
    rep, out = _run(tmp_path, make_video, make_wav, None, "No chest pain. I have a mild cough.")
    data = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert data["fusion"]["level"] != "ALERTA"
    assert any(t["negated"] for t in data["audio"]["terms"])


def test_requires_some_input():
    with pytest.raises(ValueError):
        run()
