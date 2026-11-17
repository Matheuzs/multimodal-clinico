# Resultados da análise de vídeo (rascunho para o relatório técnico)

Exercício: Ex1 do REHAB24-6 (abdução lateral do braço direito esticado), câmera 17, 30 fps.
Método: MediaPipe Pose 0.10.14 (1 de cada 3 quadros) -> ângulos de ombro/cotovelo -> segmentação de
repetições -> comparação com a linha de base da própria pessoa (mediana das 3 primeiras repetições).
Reprodução: `python -m scripts.eval_rehab PM_001 PM_012 PM_000 PM_023`

| Vídeo | Pessoa | Uso | Erradas sinalizadas | Corretas sem alarme |
|---|---|---|---|---|
| PM_001 | 1 | calibração dos limiares | 5/5 | 5/5 |
| PM_012 | 2 | calibração dos limiares | 5/5 | 5/5 |
| PM_000 | 1 | só repetições corretas | n/a | 4/5 |
| PM_023 | 3 | **validação (vídeo não usado para calibrar)** | 4/5 | 4/5 |

Na validação (PM_023): 8 de 10 repetições classificadas de acordo com o rótulo.
- Corretas sinalizadas por engano: rep. 1 (velocidade 69°/s contra mediana 118°/s da linha de base).
  Em PM_000 e PM_023 a 1ª repetição foi a atípica (possível aquecimento).
- Errada não detectada: rep. 6 (pico 140° contra linha de base ~131°, +7%, abaixo do limiar de 10%).

## Limitações
- Limiares relativos escolhidos olhando PM_001 e PM_012; só PM_023 é validação independente (1 vídeo, 10 repetições).
- O dataset não define o critério de "execução incorreta" e o erro varia por pessoa.
- A linha de base assume que as primeiras repetições da sessão estão corretas (nos vídeos do dataset, estão).
- Amplitude absoluta não generaliza (corretas: ~135° na pessoa 1, ~118° na pessoa 2, ~170° em PM_000).
- Não foi ajustado nenhum parâmetro depois de ver o PM_023.
