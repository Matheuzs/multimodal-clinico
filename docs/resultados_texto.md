# Termos críticos e sentimento (rascunho para o relatório técnico)

Entrada: fala do paciente (transcrição humana do PriMock57 e saída do Whisper). Dicionário de termos
em inglês, por categoria e severidade (`configs/config.yaml`, `audio.critical_terms`), com detecção de
negação (negador até 4 palavras antes do termo, na mesma frase) e citação da frase que gerou o alerta.
Reprodução: `python -m scripts.eval_text --source human --analyzer mock`

## Comparação com as notas clínicas do PriMock57 (5 consultas, a olho)
| Consulta | Queixa / destaques da nota | Detectado (positivo) | Só negado |
|---|---|---|---|
| 01 | diarreia, dor LLQ, febre | fadiga, gi (diarreia), vômito, dor | febre |
| 02 | pele seca e com coceira, eczema | coceira, dor | |
| 03 | cefaleia, aura visual, fotofobia | vômito, cefaleia, dor | febre |
| 04 | dor de garganta, tosse, fadiga | fadiga, febre, tosse, dor | |
| 05 | dor abdominal baixa, queimação | dor, náusea | dispneia, tosse, vômito |

A negação evitou três falsos alertas (febre na 03, "short of breath" na 05, vômito na 05); nenhuma consulta
gerou alerta de severidade alta, o que é coerente com casos benignos.

## Limitações
- Dicionário em inglês e escrito por nós; a cobertura foi ajustada depois de ver estas 5 consultas
  (acrescentamos "tired" e "pain"), então o quadro acima não é uma avaliação independente.
- Não há rótulo por frase: a comparação com as notas é qualitativa, sem precisão/recall.
- Falhas conhecidas: sintoma de outra pessoa conta ("child was vomiting" na 01; mantido de propósito,
  pois é mais seguro gerar um alerta a mais do que perder um sintoma crítico); referência temporal
  ("fever on first day, nil since" na 01 sai como negado); sinônimos fora do dicionário (aura visual, queimação).
- Sentimento: o modelo padrão (SST-2, aula 06) mede tom, não humor clínico; "I have pain" sai negativo.
  Só gera achado de severidade baixa e o limiar não foi validado.
