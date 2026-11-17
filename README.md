# Tech Challenge Fase 4 (IADT): análise multimodal clínica (MVP)

MVP com **análise de imagem/vídeo** (MediaPipe Pose), **análise de áudio**
(Whisper local, Librosa, termos críticos e sentimento) e **fusão/relatório**.
O módulo de detecção de anomalias em séries temporais está fora do escopo.

Datasets: **REHAB24-6** (vídeo) e **PriMock57** (áudio). Execução local; Azure (F0) é opcional via `USE_AZURE=true`.

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
