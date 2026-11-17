# Métricas acústicas (rascunho para o relatório técnico)

Entrada: canal do paciente do PriMock57 (um arquivo por pessoa); o canal do médico serve para separar
pausas do paciente de trocas de turno. Reprodução: `python -m scripts.eval_acoustic`.

## Detector de fala (VAD por energia)
Validado contra os intervalos de fala anotados por humanos (TextGrid), 5 consultas, quadros de 10 ms:
precisão 0,95-0,99 e F1 0,86-0,91 com `top_db=35`. O recall (0,76-0,87) é subestimado porque os
intervalos humanos incluem pausas dentro da fala.

## Consultas 01-05 (atores saudáveis)
| | fala (s) | pausas | longas (>=1,5 s) | pausa % | palavras/s | pausas longas/min |
|---|---|---|---|---|---|---|
| paciente | 159-222 | 19-79 | 3-9 | 8-23 | 2,66-3,46 | 1,0-1,9 |
| médico | 249-311 | 46-66 | 1-19 | 8-22 | 3,08-3,53 | 0,2-3,1 |

## Alertas (fala lenta, pausas longas, muita pausa, queda de energia)
Limiares escolhidos a priori, um pouco além da faixa acima. Nenhum paciente simulado dispara alerta.
## Limitações
- Sem rótulos de fadiga/dispneia no dataset: os indicadores não foram validados clinicamente;
  só foi verificado que disparam em sinais sintéticos (testes) e que não disparam em fala típica.
- Ritmo de fala usa a contagem de palavras da transcrição (humana aqui; Whisper no pipeline real).
- Energia é relativa ao próprio arquivo (percentil 95), não ao microfone; só a tendência é usada.
- Atores falando em inglês britânico, canais separados: em áudio misto não dá para descontar o médico.
