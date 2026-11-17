"""Transcrição de fala para texto, com provedores intercambiáveis.

  local_whisper  Whisper rodando local (padrão; sem custo, sem rede após baixar o modelo)
  google         SpeechRecognition.recognize_google (aula 04; API web gratuita, não oficial)
  azure          Azure Speech to Text, camada F0 (implementado na Task 10)
  mock           lê a transcrição humana (TextGrid) ao lado do áudio; só para desenvolvimento/testes

Escolha via variável TRANSCRIBER (.env) ou get_transcriber("nome").
"""
from __future__ import annotations

import os
import wave
from dataclasses import dataclass, field
from pathlib import Path

from src.audio.textgrid import read_textgrid
from src.config import ROOT, load_config, use_azure


@dataclass
class Transcript:
    text: str
    provider: str
    language: str
    segments: list[tuple[float, float, str]] = field(default_factory=list)  # (início_s, fim_s, texto)


class TranscriptionError(RuntimeError):
    pass


class Transcriber:
    name = "base"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or load_config()
        self.language = self.cfg["audio"]["language"]

    def transcribe(self, wav_path: str | Path) -> Transcript:
        raise NotImplementedError


class MockTranscriber(Transcriber):
    """Procura <nome>.TextGrid em data/raw/primock57/transcripts (ou na pasta `transcripts_dir`)."""
    name = "mock"

    def __init__(self, cfg: dict | None = None, transcripts_dir: str | Path | None = None):
        super().__init__(cfg)
        self.dir = Path(transcripts_dir) if transcripts_dir else ROOT / "data/raw/primock57/transcripts"

    def transcribe(self, wav_path) -> Transcript:
        grid = self.dir / (Path(wav_path).stem + ".TextGrid")
        if not grid.is_file():
            raise TranscriptionError(f"mock: transcrição de referência não encontrada: {grid}")
        segs = read_textgrid(grid)
        return Transcript(" ".join(t for _, _, t in segs), self.name, self.language, segs)


WHISPER_SR = 16000


def read_audio_16k(path: str | Path):
    """Lê um WAV, converte para mono float32 em 16 kHz (o formato do Whisper), sem ffmpeg."""
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != WHISPER_SR:
        import soxr
        mono = soxr.resample(mono, sr, WHISPER_SR)
    return np.ascontiguousarray(mono, dtype="float32")


class LocalWhisperTranscriber(Transcriber):
    """openai-whisper (requirements-audio.txt). O modelo é baixado na primeira execução."""
    name = "local_whisper"

    def __init__(self, cfg: dict | None = None, model=None):
        super().__init__(cfg)
        self._model = model          # injetável nos testes

    def _load(self):
        if self._model is None:
            try:
                import whisper
            except ImportError as e:
                raise TranscriptionError(
                    "openai-whisper não instalado: pip install -r requirements-audio.txt") from e
            self._model = whisper.load_model(self.cfg["audio"]["whisper_model"])
        return self._model

    def transcribe(self, wav_path) -> Transcript:
        model = self._load()
        # WAV: lido direto (sem ffmpeg). Outros formatos: o Whisper chama o ffmpeg.
        audio = read_audio_16k(wav_path) if Path(wav_path).suffix.lower() == ".wav" else str(wav_path)
        res = model.transcribe(audio, language=self.language, fp16=False)
        segs = [(float(s["start"]), float(s["end"]), s["text"].strip()) for s in res.get("segments", [])]
        return Transcript(res["text"].strip(), self.name, self.language, segs)


class GoogleTranscriber(Transcriber):
    """SpeechRecognition + recognize_google, como na aula 04. Processa em blocos de 30 s (WAV)."""
    name = "google"
    CHUNK_S = 30

    def __init__(self, cfg: dict | None = None, recognizer=None):
        super().__init__(cfg)
        self._rec = recognizer

    def _recognizer(self):
        if self._rec is None:
            try:
                import speech_recognition as sr
            except ImportError as e:
                raise TranscriptionError("SpeechRecognition não instalado (requirements-audio.txt)") from e
            self._rec = sr.Recognizer()
        return self._rec

    def transcribe(self, wav_path) -> Transcript:
        import speech_recognition as sr
        rec = self._recognizer()
        with wave.open(str(wav_path), "rb") as w:
            total = w.getnframes() / w.getframerate()
        lang = {"en": "en-US", "pt": "pt-BR"}.get(self.language, self.language)
        segs, offset = [], 0.0
        with sr.AudioFile(str(wav_path)) as src:
            while offset < total:
                audio = rec.record(src, duration=self.CHUNK_S)
                try:
                    text = rec.recognize_google(audio, language=lang)
                except sr.UnknownValueError:
                    text = ""
                except sr.RequestError as e:
                    raise TranscriptionError(f"Google Speech indisponível: {e}") from e
                if text:
                    segs.append((offset, min(offset + self.CHUNK_S, total), text))
                offset += self.CHUNK_S
        return Transcript(" ".join(s[2] for s in segs), self.name, self.language, segs)


class AzureTranscriber(Transcriber):
    name = "azure"

    def transcribe(self, wav_path) -> Transcript:
        raise NotImplementedError("Azure Speech será implementado na Task 10 (camada F0, com flag).")


_REGISTRY = {c.name: c for c in (MockTranscriber, LocalWhisperTranscriber, GoogleTranscriber, AzureTranscriber)}


def get_transcriber(name: str | None = None, cfg: dict | None = None) -> Transcriber:
    cfg = cfg or load_config()
    name = (name or os.getenv("TRANSCRIBER") or "local_whisper").strip().lower()
    if name not in _REGISTRY:
        raise ValueError(f"Transcritor desconhecido: {name!r}. Opções: {sorted(_REGISTRY)}")
    if name == "azure" and not use_azure():
        raise TranscriptionError("Azure desligado: defina USE_AZURE=true no .env para usar este provedor.")
    return _REGISTRY[name](cfg)
