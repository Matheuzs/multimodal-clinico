# Roteiro do vídeo de demonstração (alvo: 11–12 min; limite do desafio: 15 min)

Antes de gravar: `source .venv/bin/activate`, terminal com fonte grande, `outputs/report/` já gerado como plano B, abas abertas: `report.html`, vídeo anotado, `configs/config.yaml`, `docs/relatorio_tecnico.md`.

| Tempo | Cena | O que mostrar / falar |
|---|---|---|
| 0:00–1:00 | Abertura | Problema: apoiar a equipe médica analisando movimento (vídeo) e fala (áudio). Escopo: MVP sem detecção de anomalias em sinais vitais. |
| 1:00–2:30 | Arquitetura | Diagrama da seção 2 do relatório; um módulo por modalidade; `Finding` comum; tudo local e sem custo. Explicar a troca OpenPose→MediaPipe e que o Azure foi dispensado (arquitetura permite plugar). |
| 2:30–3:30 | Dados | REHAB24-6 (vídeo, rótulos correta/incorreta) e PriMock57 (consultas simuladas, canais separados). |
| 3:30–6:00 | Vídeo | Abrir o vídeo anotado: esqueleto, ângulos; explicar repetições e linha de base da própria pessoa; mostrar repetições 1–5 normais e 6–10 sinalizadas (cotovelo, velocidade, amplitude). Citar a validação: 8/10 no vídeo independente. |
| 6:00–8:30 | Áudio | Whisper `small` (WER 17,7%); indicadores de pausa e ritmo; termos críticos com negação ("no fever" não alerta); sentimento. Mostrar a tabela de termos no relatório. |
| 8:30–10:00 | Fusão e alerta | Rodar `python -m src.pipeline ... --annotate`; mostrar saída "Nível: ALERTA"; explicar score por regra distinta, convergência e regra de alta severidade. Mostrar a mensagem para a equipe médica. |
| 10:00–11:00 | Relatório | Abrir `report.html`: nível, evidências, tabela de repetições, indicadores acústicos. |
| 11:00–12:00 | Qualidade e limites | `python -m pytest -q` (90 testes); limitações honestas: poucos dados, limiares provisórios, vídeo e áudio de pessoas diferentes nesta demo, não é diagnóstico. |
| 12:00–12:30 | Encerramento | Trabalhos futuros e link do repositório/relatório. |

Dicas: rode o pipeline com antecedência (o vídeo leva ~20 s, o Whisper cerca de 30 s) e grave só a saída; deixe claro, na fala, que o par vídeo+áudio é uma demonstração do mecanismo; envie o vídeo ao YouTube/Vimeo como "não listado" e cole o link no relatório.
