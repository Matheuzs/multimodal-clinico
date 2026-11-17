# Fusão multimodal (Task 11)

`src.fusion.fusion.fuse(vision_reps, audio_findings)` combina os achados das duas modalidades em um nível de risco.

| Passo | Regra | Por quê |
|---|---|---|
| 1 | Cada modalidade soma o peso da maior severidade de cada **regra distinta** (low 1, medium 2, high 3) | 10 repetições com o mesmo desvio são um achado, não dez |
| 2 | Regras de visão só contam se ≥ 20% das repetições as dispararam | ignora ruído isolado de pose |
| 3 | +1 de bônus quando visão **e** áudio têm achado ≥ medium | convergência entre fontes independentes |
| 4 | ALERTA se houver achado `high`, ou se score ≥ 4 **com convergência** entre vídeo e áudio; ATENÇÃO se score ≥ 2; senão NORMAL | achado crítico (ex.: dor no peito) não depende de somar pontos; várias regras leves de uma só fonte (teste real: consulta de diarreia com 4 termos, score 6) davam alerta demais |

Todos os valores estão em `configs/config.yaml` (`fusion:`). Cada evidência do resultado cita a modalidade, a regra, a severidade e, na visão, as repetições envolvidas.

Limitações para o relatório:
- Pesos, limiares e bônus são **provisórios e sem validação clínica**; os dados disponíveis não têm rótulo de desfecho.
- Visão (reabilitação, REHAB24-6) e áudio (consultas, PriMock57) vêm de **datasets diferentes**: a convergência é demonstrada com um par vídeo+áudio escolhido para a demo, não com dados de um mesmo paciente.
- O alerta é apoio à decisão; sempre indica avaliação profissional.
