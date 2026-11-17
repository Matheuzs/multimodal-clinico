import pytest

from src.audio.metrics import normalize, wer
from src.audio.textgrid import clean_text, read_textgrid, reference_text
from src.audio.transcriber import (GoogleTranscriber, LocalWhisperTranscriber, MockTranscriber,
                                   TranscriptionError, get_transcriber)
from src.config import ROOT

GRID = '''File type = "ooTextFile"
Object class = "TextGrid"
item []:
    item [1]:
        class = "IntervalTier"
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 2.5
            text = ""
        intervals [2]:
            xmin = 2.5
            xmax = 7.25
            text = "<UNSURE>Um</UNSURE>, I have chest pain <UNIN/> since "yesterday"."
        intervals [3]:
            xmin = 7.25
            xmax = 9
            text = "<INAUDIBLE_SPEECH/>"
'''


@pytest.fixture
def grid(tmp_path):
    p = tmp_path / "x_patient.TextGrid"
    p.write_text(GRID, encoding="utf-8")
    return p


# ---- TextGrid
def test_clean_text_removes_tags_keeps_content():
    assert clean_text("<UNSURE>Um</UNSURE>, ok <UNIN/> yes") == "Um , ok yes"


def test_read_textgrid_skips_empty_and_untext(grid):
    segs = read_textgrid(grid)
    assert len(segs) == 1 and segs[0][:2] == (2.5, 7.25)
    assert "chest pain" in segs[0][2]


# ---- WER
def test_wer_identical_and_normalization():
    assert wer("Chest pain, since yesterday.", "chest PAIN since yesterday")["wer"] == 0


def test_wer_ignores_fillers():
    assert wer("um I uh have pain", "I have pain")["wer"] == 0


def test_wer_counts_errors():
    r = wer("i have chest pain", "i had chest pain today")
    assert (r["substitutions"], r["deletions"], r["insertions"]) == (1, 0, 1)
    assert r["wer"] == pytest.approx(2 / 4)


def test_wer_deletion_and_empty():
    assert wer("a b c d", "a b")["deletions"] == 2
    assert wer("", "")["wer"] == 0 and wer("a", "")["wer"] == 1.0


def test_normalize_apostrophes():
    assert normalize("I've got it’s") == ["i've", "got", "it's"]


# ---- transcritores
def test_mock_reads_human_transcript(grid, make_wav):
    wav = make_wav("x_patient.wav")
    t = MockTranscriber(transcripts_dir=grid.parent).transcribe(wav)
    assert t.provider == "mock" and "chest pain" in t.text and len(t.segments) == 1


def test_mock_missing_reference(make_wav, tmp_path):
    with pytest.raises(TranscriptionError):
        MockTranscriber(transcripts_dir=tmp_path).transcribe(make_wav())


def test_whisper_with_injected_model(make_wav):
    class Fake:
        def transcribe(self, path, **kw):
            assert kw["language"] == "en"
            return {"text": " I feel dizzy. ", "segments": [{"start": 0, "end": 2, "text": " I feel dizzy. "}]}
    t = LocalWhisperTranscriber(model=Fake()).transcribe(make_wav())
    assert t.text == "I feel dizzy." and t.segments == [(0.0, 2.0, "I feel dizzy.")]


def test_whisper_receives_mono_16k_array_without_ffmpeg(make_wav):
    pytest.importorskip("soundfile")
    pytest.importorskip("soxr")
    import numpy as np
    seen = {}
    class Fake:
        def transcribe(self, audio, **kw):
            seen["audio"] = audio
            return {"text": "ok", "segments": []}
    LocalWhisperTranscriber(model=Fake()).transcribe(make_wav(seconds=2, sr=44100, channels=2))
    a = seen["audio"]
    assert isinstance(a, np.ndarray) and a.dtype == np.float32 and a.ndim == 1
    assert len(a) == pytest.approx(2 * 16000, abs=50)


def test_google_chunks_and_unknown_value(make_wav):
    sr = pytest.importorskip("speech_recognition")
    class Fake:
        calls = 0
        def record(self, src, duration=None):
            return object()
        def recognize_google(self, audio, language=None):
            Fake.calls += 1
            if Fake.calls == 2:
                raise sr.UnknownValueError()
            assert language == "en-US"
            return f"part{Fake.calls}"
    t = GoogleTranscriber(recognizer=Fake()).transcribe(make_wav(seconds=70))
    assert Fake.calls == 3 and t.text == "part1 part3"
    assert [s[0] for s in t.segments] == [0.0, 60.0]


# ---- fábrica
def test_factory_default_and_unknown(monkeypatch):
    monkeypatch.delenv("TRANSCRIBER", raising=False)
    assert get_transcriber().name == "local_whisper"
    with pytest.raises(ValueError):
        get_transcriber("xyz")


def test_factory_env(monkeypatch):
    monkeypatch.setenv("TRANSCRIBER", "mock")
    assert get_transcriber().name == "mock"
    with pytest.raises(ValueError):
        get_transcriber("azure")


# ---- dados reais
def test_real_primock_mock_roundtrip_wer_zero():
    wav = ROOT / "data/raw/primock57/audio/day1_consultation01_patient.wav"
    if not wav.exists():
        pytest.skip("PriMock57 não baixado")
    hyp = MockTranscriber().transcribe(wav).text
    ref = reference_text(ROOT / "data/raw/primock57/transcripts/day1_consultation01_patient.TextGrid")
    assert len(normalize(ref)) > 100 and wer(ref, hyp)["wer"] == 0
