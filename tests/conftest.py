import wave

import cv2
import numpy as np
import pytest


@pytest.fixture
def make_video(tmp_path):
    def _make(name="v.mp4", seconds=2, fps=10, size=(64, 48)):
        p = tmp_path / name
        w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        for i in range(int(seconds * fps)):
            w.write(np.full((size[1], size[0], 3), i % 255, np.uint8))
        w.release()
        return p
    return _make


@pytest.fixture
def make_wav(tmp_path):
    def _make(name="a.wav", seconds=1.0, sr=16000, channels=1):
        p = tmp_path / name
        n = int(seconds * sr)
        data = (np.sin(np.linspace(0, 440 * 2 * np.pi * seconds, n)) * 3000).astype("<i2")
        if channels > 1:
            data = np.repeat(data[:, None], channels, axis=1).ravel()
        with wave.open(str(p), "wb") as w:
            w.setnchannels(channels); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes(data.tobytes())
        return p
    return _make
