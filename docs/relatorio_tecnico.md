# Relatório Técnico — Tech Challenge Fase 4 (IADT)
**Análise multimodal de vídeo e áudio com fusão e alerta para a equipe médica**
Autor: Matheus · Repositório: _(inserir link do GitHub)_ · Vídeo de demonstração: _(inserir link)_

## 1. Escopo e decisões de projeto
O projeto implementa um MVP que analisa **vídeo** (execução de exercício de reabilitação) e **áudio** (fala do paciente em consulta), combina os achados em um **relatório com nível de risco** e gera um alerta para a equipe médica.

Decisões tomadas e justificativas:
| Decisão | Justificativa |
|---|---|
| Sem módulo de detecção de anomalias em séries temporais/sinais vitais/prescrições | Fora do escopo acordado para este MVP; o foco são as modalidades de imagem/vídeo e áudio. |
| Análise de **vídeo** em vez de imagem estática | Um movimento só pode ser avaliado ao longo do tempo (repetições, velocidade, amplitude). |
| **MediaPipe Pose** no lugar do OpenPose | Roda em CPU/Mac sem compilar dependências, sem GPU e sem custo; fornece 33 pontos do corpo, suficientes para ângulos de ombro e cotovelo. |
| Modelos locais e de código aberto | Custo zero e reprodutibilidade; nenhum serviço pago é necessário. |
| **Sem Azure** | O enunciado cita o Azure, mas a integração foi dispensada por escolha de projeto: tudo roda localmente, sem chaves e sem consumo de cota. A arquitetura mantém transcrição e sentimento atrás de interfaces (`Transcriber`, `SentimentAnalyzer`), então um provedor em nuvem poderia ser acrescentado sem alterar o restante. |
| Um dataset por modalidade | REHAB24-6 (vídeo) e PriMock57 (áudio); ver seção 3. |

## 2. Fluxo multimodal
```
 vídeo ──► ingestão ──► MediaPipe Pose ──► ângulos ──► repetições ──► regras vs. linha de base ─┐
                                                                                                │
 áudio ──► ingestão ──► Whisper (ASR) ──► texto ──► termos críticos + negação + sentimento ─────┼─► FUSÃO ─► relatório
          (paciente)  └─► VAD/energia ──► pausas, ritmo de fala, tendência de energia ──────────┘   (score, nível, evidências)
          (+ canal do médico, opcional, para descontar trocas de turno)                          HTML / Markdown / JSON
```
Todos os módulos produzem o mesmo tipo de objeto, `Finding` (regra, severidade baixa/média/alta, mensagem, valor, limiar), o que permite a fusão sem conhecer os detalhes de cada modalidade.

Estrutura do código: `src/ingestion` (validação), `src/vision` (pose e regras), `src/audio` (transcrição, VAD, acústica, texto), `src/fusion`, `src/report`, `src/pipeline.py` (execução completa). Parâmetros em `configs/config.yaml`; 90 testes automatizados em `tests/`. Evidência completa de cada etapa em `docs/resultados_detalhados.md`.

## 3. Datasets
| Modalidade | Dataset | Uso | Observação |
|---|---|---|---|
| Vídeo | **REHAB24-6** | Exercício 1 (abdução lateral do braço direito esticado), câmera 17, 30 fps; rótulos de repetição correta/incorreta em `Segmentation.csv` | Baixamos só alguns vídeos (extração parcial do arquivo zip remoto). |
| Áudio | **PriMock57** | Consultas médicas simuladas, com áudio do paciente e do médico em arquivos separados e transcrição humana (TextGrid) | 5 consultas baixadas; atores, inglês britânico. |

## 4. Modelos e métodos por tipo de dado
### 4.1 Vídeo
- **MediaPipe Pose** (`mediapipe==0.10.14`, complexidade 1), 1 quadro a cada 3, imagem reduzida a 640 px de largura.
- Ângulos 2D de ombro (quadril–ombro–cotovelo) e cotovelo (ombro–cotovelo–punho), calculados em pixels, com descarte de pontos de baixa visibilidade.
- **Segmentação de repetições** pela série do ângulo do ombro direito (subida acima de 40° até pico ≥ 60° e retorno abaixo de 30°); repetições truncadas no início do vídeo são descartadas.
- **Regras relativas à linha de base da própria pessoa** (mediana das 3 primeiras repetições): pico do ombro 10% menor (`desvio_amplitude`), velocidade 35% menor (`desvio_velocidade`), cotovelo 10° diferente (`desvio_cotovelo`); além de regras absolutas de compensação do outro ombro e qualidade de detecção. Uma regra de amplitude absoluta foi **descartada** porque não generalizou entre pessoas.

### 4.2 Áudio
- **Transcrição:** Whisper local (`openai-whisper`), modelo `small`; alternativa `recognize_google` (aula 04) e um provedor `mock` que lê a transcrição humana. WAV lido sem ffmpeg (soundfile + soxr, mono 16 kHz).
- **Indicadores acústicos:** detector de fala por energia (RMS em janelas de 25 ms), pausas (longas ≥ 1,5 s), palavras por segundo de fala, tendência de energia (último terço menos primeiro terço). Com o canal do médico, pausas coincidentes com a fala do outro não contam como pausa do paciente.
- **Texto:** dicionário de 15 categorias de termos críticos em inglês com severidade e **detecção de negação**; **sentimento** com `pipeline('sentiment-analysis')` da biblioteca `transformers` (modelo padrão DistilBERT SST-2, aula 06).

### 4.3 Fusão
Regras em `docs/resultados_detalhados.md` (seção 5): cada modalidade soma o peso (baixa 1, média 2, alta 3) da maior severidade de cada **regra distinta**; regras de vídeo só valem se ≥ 20% das repetições as dispararam; bônus de +1 quando vídeo e áudio têm achado ≥ média. **ALERTA** se houver achado de severidade alta, ou score ≥ 4 com convergência entre as duas modalidades; **ATENÇÃO** se score ≥ 2; senão NORMAL.

## 5. Resultados
### 5.1 Vídeo (REHAB24-6, Ex1, câmera 17)
| Vídeo | Pessoa | Uso | Erradas sinalizadas | Corretas sem alarme |
|---|---|---|---|---|
| PM_001 | 1 | calibração | 5/5 | 5/5 |
| PM_012 | 2 | calibração | 5/5 | 5/5 |
| PM_000 | 1 | só corretas | n/a | 4/5 |
| PM_023 | 3 | **validação independente** | 4/5 | 4/5 |

Na validação independente, 8 de 10 repetições coincidiram com o rótulo. Falha 1: a 1ª repetição correta foi sinalizada (velocidade menor que a da linha de base, possivelmente aquecimento). Falha 2: uma repetição errada com pico só 7% acima da linha de base ficou abaixo do limiar de 10%.

### 5.2 Transcrição (PriMock57, 3 consultas, 1584 palavras)
| Whisper | WER médio | tempo por consulta |
|---|---|---|
| base | 25,1% | 9–13 s |
| **small** | **17,7%** | 21–29 s |

O `base` trocou termos médicos (itching→teaching, eczema→exam, diarrhea→diaries). Na lista de 19 ocorrências de termos clínicos, o `base` acertou 6 e o `small` 19 (amostra pequena e lista definida depois de ver os erros). Por isso o padrão é `small`.

### 5.3 Indicadores acústicos (5 consultas)
- Detector de fala contra os intervalos humanos: precisão 0,95–0,99 e F1 0,86–0,91.
- Paciente: 2,66–3,46 palavras/s, 8–23% de pausa, 1,0–1,9 pausas longas por minuto. Nenhum dos atores saudáveis disparou alerta acústico, como esperado.

### 5.4 Termos críticos
Nas 5 consultas o detector capturou a queixa principal em todas (diarreia, coceira, cefaleia, tosse, dor abdominal) e a negação evitou 3 falsos alertas ("no fever", "not short of breath"). Nenhuma consulta gerou achado de severidade alta, coerente com casos benignos. Comparação qualitativa com as notas clínicas, não é métrica de precisão/recall.

### 5.5 Exemplos de desvio e de alerta (execução completa)
Comando: `python -m src.pipeline --video PM_001-Camera17-30fps.mp4 --audio day1_consultation01_patient.wav --other-audio day1_consultation01_doctor.wav --annotate`

| Item | Resultado |
|---|---|
| Vídeo | 10 repetições; as 5 primeiras (corretas) sem achados; as repetições 6 a 10 (erradas) sinalizadas: cotovelo ~15° mais esticado que a referência (169° contra 153°), velocidade −36%, amplitude −13% |
| Áudio | termos de fraqueza (média), vômito (média), diarreia e dor (baixa); fala a 2,7 palavras/s, sem alerta acústico |
| Fusão | score 11 (vídeo 4, áudio 6, convergência 1) → **ALERTA**, com mensagem: "ALERTA à equipe médica … Vídeo e áudio apontam desvios simultâneos. Requer avaliação profissional." |

**Atenção:** o vídeo e o áudio vêm de pessoas diferentes e de datasets diferentes; o par foi escolhido para demonstrar o mecanismo de fusão, não representa um caso clínico real.

### 5.6 Calibração da fusão com dado real
Na primeira execução só com áudio, uma consulta benigna de diarreia chegou a ALERTA (score 6) apenas por somar quatro termos leves/médios. A regra foi corrigida: score alto só gera ALERTA com convergência entre as modalidades (ou achado de severidade alta). O mesmo caso passou a ATENÇÃO.

## 6. Limitações
- **Validação estatística fraca:** limiares de vídeo calibrados em dois vídeos; só um vídeo independente (10 repetições). Resultados indicam viabilidade, não desempenho clínico.
- **Sem rótulos clínicos** para áudio: limiares acústicos escolhidos a priori e dicionário ajustado depois de ver as 5 consultas; sentimento (SST-2) mede tom, não humor clínico.
- **Modalidades de pessoas diferentes** no exemplo de fusão; pesos e limiares da fusão são provisórios.
- Falhas conhecidas do texto: sintoma de terceiros conta como do paciente (mantido por segurança), referência temporal não é interpretada.
- Vídeo: ângulos 2D dependem do enquadramento da câmera; a linha de base assume que as primeiras repetições estão corretas.
- Dados simulados (atores, inglês britânico), sem pacientes reais; nenhum dado pessoal foi coletado.
- O sistema é apoio à triagem e **não substitui avaliação profissional**.

## 7. Como reproduzir
```
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-vision.txt -r requirements-audio.txt
bash scripts/download_data.sh all
python -m pytest -q
python -m src.pipeline --video <video.mp4> --audio <paciente.wav> --other-audio <medico.wav> --annotate
```
Avaliações individuais: `scripts/eval_rehab.py`, `scripts/eval_asr.py`, `scripts/eval_acoustic.py`, `scripts/eval_text.py`.

## 8. Trabalhos futuros
Validar limiares com mais pessoas e mais exercícios; dataset de áudio com rótulos clínicos; dicionário em português; estimativa de pose em 3D ou com várias câmeras; integração opcional com serviços em nuvem; mais modalidades (sinais vitais), hoje fora do escopo.
