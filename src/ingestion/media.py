"""Ingestão e validação de mídia (vídeo, imagem, áudio).

Só valida e descreve o arquivo; o processamento fica em vision/ e audio/.
Áudio: WAV via biblioteca padrão; outros formatos exigem `soundfile`
(requirements-audio.txt).
"""
from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.config import load_config


class IngestionError(ValueError):
    """Arquivo ausente, formato não suportado, corrompido ou fora dos limites."""


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    kind: str                 # "video" | "image" | "audio"
    duration_s: float | None  # None para imagem
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frames: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    warnings: tuple[str, ...] = ()


def detect_kind(path: Path, cfg: dict | None = None) -> str:
    cfg = cfg or load_config()
    ing = cfg["ingestion"]
    ext = path.suffix.lower()
    if ext in ing["video_exts"]:
        return "video"
    if ext in ing["image_exts"]:
        return "image"
    if ext in ing["audio_exts"]:
        return "audio"
    raise IngestionError(f"Formato não suportado: '{ext}' ({path.name})")


def _check_exists(path: Path) -> Path:
    path = Path(path)
    if not path.is_file():
        raise IngestionError(f"Arquivo não encontrado: {path}")
    if path.stat().st_size == 0:
        raise IngestionError(f"Arquivo vazio: {path}")
    return path


def validate_video(path: Path, cfg: dict | None = None) -> MediaInfo:
    cfg = cfg or load_config()
    path = _check_exists(path)
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise IngestionError(f"Não foi possível abrir o vídeo: {path.name}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, _ = cap.read()
        if not ok or fps <= 0 or frames <= 0:
            raise IngestionError(f"Vídeo sem frames legíveis: {path.name}")
    finally:
        cap.release()
    duration = frames / fps
    limit = cfg["ingestion"]["max_video_seconds"]
    if duration > limit:
        raise IngestionError(
            f"Vídeo com {duration:.0f}s excede o limite de {limit}s ({path.name})"
        )
    return MediaInfo(path, "video", duration, width, height, fps, frames)


def validate_image(path: Path, cfg: dict | None = None) -> MediaInfo:
    path = _check_exists(path)
    img = cv2.imread(str(path))
    if img is None:
        raise IngestionError(f"Imagem inválida ou corrompida: {path.name}")
    h, w = img.shape[:2]
    return MediaInfo(path, "image", None, w, h)


def validate_audio(path: Path, cfg: dict | None = None) -> MediaInfo:
    cfg = cfg or load_config()
    path = _check_exists(path)
    if path.suffix.lower() != ".wav":
        raise IngestionError(
            f"Validação implementada para WAV; '{path.suffix}' não é suportado."
        )
    try:
        with wave.open(str(path), "rb") as w:
            sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
    except (wave.Error, EOFError) as e:
        raise IngestionError(f"WAV inválido ({path.name}): {e}") from e
    if n == 0 or sr == 0:
        raise IngestionError(f"Áudio sem amostras: {path.name}")
    warnings = []
    target = cfg["ingestion"]["target_audio_sr"]
    if sr != target:
        warnings.append(f"sample rate {sr} Hz difere do alvo {target} Hz (será reamostrado)")
    if ch > 1:
        warnings.append(f"{ch} canais (será convertido para mono)")
    return MediaInfo(path, "audio", n / sr, sample_rate=sr, channels=ch,
                     warnings=tuple(warnings))


_VALIDATORS = {"video": validate_video, "image": validate_image, "audio": validate_audio}


def ingest(path: str | Path, cfg: dict | None = None) -> MediaInfo:
    """Detecta o tipo pela extensão e valida o arquivo."""
    cfg = cfg or load_config()
    path = Path(path)
    return _VALIDATORS[detect_kind(path, cfg)](path, cfg)
