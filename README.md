# Tech Challenge Fase 4 (IADT): análise multimodal clínica (MVP)

MVP com **análise de imagem/vídeo** (MediaPipe Pose), **análise de áudio**
(Whisper local, Librosa, termos críticos e sentimento) e **fusão/relatório**.
O módulo de detecção de anomalias em séries temporais está fora do escopo.

Datasets: **REHAB24-6** (vídeo) e **PriMock57** (áudio). Execução 100% local, sem serviços pagos.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
pytest
```
Dependências pesadas: `requirements-vision.txt` e `requirements-audio.txt` (instaladas nas tasks seguintes).

## Estrutura
`src/{ingestion,vision,audio,cloud,fusion,report}`, `configs/`, `scripts/`, `docs/`, `tests/`, `data/`, `outputs/`.

## Execução completa (Task 12)
```
python -m src.pipeline --video data/raw/rehab24/<video>.mp4 --start 0 --end 900 \
  --audio data/raw/primock57/audio/day1_consultation01_patient.wav \
  --other-audio data/raw/primock57/audio/day1_consultation01_doctor.wav --annotate
```
Gera `outputs/report/report.html`, `report.md` e `report.json` (e o vídeo anotado com `--annotate`). `--video` e `--audio` são opcionais entre si.
