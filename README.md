# Análise multimodal clínica — Tech Challenge Fase 4 (IADT)

MVP que analisa **vídeo** de um exercício de reabilitação e **áudio** de uma consulta médica, combina
os achados das duas fontes e gera um **relatório com nível de risco** (NORMAL, ATENÇÃO ou ALERTA) para
a equipe médica.

Tudo roda **localmente**, com modelos de código aberto: nenhuma chave de API, nenhum serviço pago.

- Relatório técnico: [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md)
- Resultados medidos e limitações: [`docs/resultados_detalhados.md`](docs/resultados_detalhados.md)
- Vídeo de demonstração: _(inserir link)_

## O que o sistema faz

| Fonte | O que extrai | Modelo / método |
|---|---|---|
| Vídeo | ângulos de ombro e cotovelo, repetições do exercício, desvios de amplitude, velocidade e postura | MediaPipe Pose |
| Áudio | transcrição da fala | Whisper local (modelo `small`) |
| Áudio | pausas, ritmo de fala, tendência de energia da voz | detector de fala por energia (NumPy) |
| Texto | termos clínicos por categoria e severidade, com detecção de negação; tom da fala | dicionário próprio + `transformers` (DistilBERT SST-2) |
| Fusão | nível de risco, score e lista de evidências | regras com pesos por severidade |

Cada módulo devolve o mesmo objeto `Finding` (regra, severidade, mensagem, valor, limiar), o que
permite somar evidências de fontes diferentes sem acoplar os módulos.

## Instalação

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-vision.txt -r requirements-audio.txt
python -m pytest -q          # 90 testes, sem baixar modelos
```

Na primeira execução real, o Whisper (~461 MB) e o modelo de sentimento (~268 MB) são baixados
automaticamente.

## Datasets

```bash
bash scripts/download_data.sh all     # ou: primock (áudio) | rehab (vídeo)
```

- **REHAB24-6** (vídeo): exercício 1, câmera 17, com rótulo de repetição correta ou incorreta.
  Baixa apenas os vídeos necessários, por extração parcial do arquivo remoto.
- **PriMock57** (áudio): consultas simuladas, com paciente e médico em arquivos separados e
  transcrição humana de referência.

Os dados ficam em `data/raw/` e não são versionados.

## Execução

```bash
python -m src.pipeline \
  --video data/raw/rehab24/videos/PM_001-Camera17-30fps.mp4 \
  --audio data/raw/primock57/audio/day1_consultation01_patient.wav \
  --other-audio data/raw/primock57/audio/day1_consultation01_doctor.wav \
  --annotate
```

Gera em `outputs/report/`: `report.html`, `report.md`, `report.json` e o vídeo com o esqueleto
desenhado. `--video` e `--audio` são opcionais entre si, então é possível analisar só uma das fontes.

Opções úteis: `--start`/`--end` (recorte em quadros), `--other-audio` (canal do médico, usado para não
contar troca de turno como pausa do paciente), `--transcriber local_whisper|google|mock`,
`--analyzer local|mock` (`mock` não baixa modelos), `--out`.

Avaliações isoladas, que reproduzem os números do relatório:

```bash
python -m scripts.eval_rehab PM_001 PM_012 PM_000 PM_023    # vídeo vs. rótulos do dataset
python -m scripts.eval_asr --provider local_whisper --model small --n 3   # WER
python -m scripts.eval_acoustic                             # pausas e ritmo de fala
python -m scripts.eval_text --source human --analyzer mock   # termos clínicos
```

## Estrutura

```
src/ingestion   validação de vídeo e áudio
src/vision      pose (MediaPipe), segmentação de repetições e regras
src/audio       transcrição, detector de fala, métricas acústicas, termos e sentimento
src/fusion      combinação dos achados e nível de risco
src/report      relatório em Markdown, HTML e JSON
src/pipeline.py execução completa (CLI)
configs/        limiares e dicionário de termos
scripts/        download dos dados e avaliações
tests/          90 testes
```

## Resultados (resumo)

- **Vídeo:** no vídeo não usado para calibrar, 8 de 10 repetições coincidiram com o rótulo do dataset
  (1 falso positivo, 1 falso negativo).
- **Transcrição:** WER de 17,7% com o modelo `small`, contra 25,1% do `base`, em 3 consultas.
- **Detector de fala:** precisão de 0,95 a 0,99 e F1 de 0,86 a 0,91 contra a anotação humana.
- **Termos clínicos:** a queixa principal foi detectada nas 5 consultas; a negação evitou 3 falsos alertas.

## Limitações

Os limiares foram calibrados em poucos vídeos e os indicadores de áudio não têm rótulo clínico para
validação. Os datasets de vídeo e de áudio são de pessoas diferentes, então o exemplo de fusão
demonstra o mecanismo, não um caso clínico real. O sistema é **apoio à triagem e não substitui
avaliação profissional**. A lista completa está no relatório técnico.

## Escopo

Implementado: análise de vídeo, análise de áudio, fusão e relatório. Fora do escopo deste MVP:
detecção de anomalias em séries temporais, sinais vitais e prescrições.
