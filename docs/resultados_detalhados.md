# Resultados detalhados e método (evidência para o relatório técnico)

Documento de apoio: reúne os números medidos, os comandos que os reproduzem e as limitações de cada
etapa. O resumo está em `relatorio_tecnico.md`.

---
## 1. Análise de vídeo (REHAB24-6)
Exercício 1 (abdução lateral do braço direito esticado), câmera 17, 30 fps.
Método: MediaPipe Pose 0.10.14 (1 de cada 3 quadros) → ângulos de ombro e cotovelo → segmentação de
repetições → comparação com a linha de base da própria pessoa (mediana das 3 primeiras repetições).
Reprodução: `python -m scripts.eval_rehab PM_001 PM_012 PM_000 PM_023`

| Vídeo | Pessoa | Uso | Erradas sinalizadas | Corretas sem alarme |
|---|---|---|---|---|
| PM_001 | 1 | calibração dos limiares | 5/5 | 5/5 |
| PM_012 | 2 | calibração dos limiares | 5/5 | 5/5 |
| PM_000 | 1 | só repetições corretas | n/a | 4/5 |
| PM_023 | 3 | **validação (não usado para calibrar)** | 4/5 | 4/5 |

Na validação (PM_023), 8 de 10 repetições coincidiram com o rótulo:
- Falso positivo: repetição 1 (velocidade 69°/s contra mediana 118°/s da linha de base). Em PM_000 e
  PM_023 a 1ª repetição foi a atípica, possivelmente aquecimento.
- Falso negativo: repetição 6 (pico 140° contra linha de base ~131°, +7%, abaixo do limiar de 10%).

Limitações:
- Limiares escolhidos olhando PM_001 e PM_012; só PM_023 é validação independente (1 vídeo, 10 repetições).
- O dataset não define o critério de "execução incorreta" e o erro varia por pessoa.
- A linha de base assume que as primeiras repetições da sessão estão corretas (nestes vídeos, estão).
- Amplitude absoluta não generaliza (corretas: ~135° na pessoa 1, ~118° na pessoa 2, ~170° em PM_000),
  por isso a regra absoluta de amplitude ficou desativada e a avaliação é relativa.
- Nenhum parâmetro foi ajustado depois de ver o PM_023.

---
## 2. Transcrição de áudio (PriMock57)
3 consultas (day1_consultation01–03, 1584 palavras de referência). Referência: transcrição humana
(TextGrid). WER com normalização: minúsculas, sem pontuação, sem interjeições (um, uh, hmm).
Reprodução: `python -m scripts.eval_asr --provider local_whisper --model small --n 3`

| Modelo Whisper | WER c. 01 | 02 | 03 | WER médio ponderado | tempo por consulta |
|---|---|---|---|---|---|
| base (139 MB) | 19,6% | 23,8% | 31,9% | 25,1% | 9–13 s |
| small (461 MB) | 13,7% | 19,2% | 19,1% | **17,7%** | 21–29 s |

Erros do `base`: termos médicos trocados (itching→teaching, eczema→exam, diarrhea→diaries,
stool→stall) e perda de falas curtas. Apenas 3 trechos de 5+ palavras foram perdidos.

Termos clínicos reconhecidos (19 ocorrências na referência humana, lista fixa de 10 termos):
`base` 6/19 (itching 0/3, eczema 0/4, diarrhea 0/2, stool 0/2); `small` 19/19. Amostra pequena e lista
escolhida após ver os erros do `base`: indica tendência, não é benchmark. Por isso o padrão é `small`.

Limitações: 3 consultas do primeiro dia, um locutor por consulta; fala simulada por atores em inglês
britânico; variações de grafia ("OK"/"okay", "six"/"6") podem contar como erro.

---
## 3. Métricas acústicas
Entrada: canal do paciente (um arquivo por pessoa); o canal do médico serve para separar pausas do
paciente de trocas de turno. Reprodução: `python -m scripts.eval_acoustic`

Detector de fala (VAD por energia), validado contra os intervalos anotados por humanos (5 consultas,
quadros de 10 ms): precisão 0,95–0,99 e F1 0,86–0,91 com `top_db=35`. O recall (0,76–0,87) é
subestimado porque os intervalos humanos incluem pausas dentro da fala.

| Consultas 01–05 | fala (s) | pausas | longas (≥1,5 s) | pausa % | palavras/s | longas/min |
|---|---|---|---|---|---|---|
| paciente | 159–222 | 19–79 | 3–9 | 8–23 | 2,66–3,46 | 1,0–1,9 |
| médico | 249–311 | 46–66 | 1–19 | 8–22 | 3,08–3,53 | 0,2–3,1 |

Limiares escolhidos a priori, um pouco além da faixa acima: nenhum paciente simulado dispara alerta,
que é o comportamento esperado para pessoas saudáveis.

Limitações:
- Sem rótulos de fadiga ou dispneia no dataset: os indicadores não foram validados clinicamente. Só foi
  verificado que disparam em sinais sintéticos (testes) e que não disparam em fala típica.
- Ritmo de fala usa a contagem de palavras da transcrição (humana aqui, Whisper no pipeline).
- Energia é relativa ao próprio arquivo (percentil 95), não ao microfone: só a tendência é usada.
- Em áudio com os dois interlocutores no mesmo canal não é possível descontar a fala do médico.

---
## 4. Termos críticos e sentimento
Dicionário de 15 categorias em inglês, com severidade (`configs/config.yaml`, `audio.critical_terms`),
detecção de negação (negador até 4 palavras antes do termo, na mesma frase) e citação da frase que
gerou o achado. Reprodução: `python -m scripts.eval_text --source human --analyzer mock`

| Consulta | Queixa / nota clínica | Detectado | Só negado |
|---|---|---|---|
| 01 | diarreia, dor abdominal baixa, febre | fadiga, gi (diarreia), vômito, dor | febre |
| 02 | pele seca e com coceira, eczema | coceira, dor | |
| 03 | cefaleia, aura visual, fotofobia | vômito, cefaleia, dor | febre |
| 04 | dor de garganta, tosse, fadiga | fadiga, febre, tosse, dor | |
| 05 | dor abdominal baixa, queimação | dor, náusea | dispneia, tosse, vômito |

A negação evitou três falsos alertas (febre na 03, "short of breath" na 05, vômito na 05). Nenhuma
consulta gerou achado de severidade alta, coerente com casos benignos.

Limitações:
- Dicionário escrito por nós e ajustado depois de ver estas 5 consultas (acrescentamos "tired" e
  "pain"), então o quadro acima não é uma avaliação independente.
- Não há rótulo por frase: a comparação com as notas clínicas é qualitativa, sem precisão/recall.
- Falhas conhecidas: sintoma de terceiros conta como do paciente ("child was vomiting" na 01, mantido
  de propósito, porque perder um sintoma crítico é pior do que um alerta extra); referência temporal
  não é interpretada ("fever on first day, nil since" sai como negado); sinônimos fora do dicionário
  (aura visual, queimação).
- Sentimento: o modelo padrão (DistilBERT SST-2) mede tom, não humor clínico ("I have pain" sai como
  negativo). Por isso gera apenas achado de severidade baixa, com limiar não validado.

---
## 5. Fusão
`src.fusion.fusion.fuse(vision_reps, audio_findings)` combina os achados em um nível de risco.

| Passo | Regra | Por quê |
|---|---|---|
| 1 | Cada modalidade soma o peso da maior severidade de cada **regra distinta** (baixa 1, média 2, alta 3) | 10 repetições com o mesmo desvio são um achado, não dez |
| 2 | Regras de vídeo só contam se ≥ 20% das repetições as dispararam | ignora ruído isolado de pose |
| 3 | +1 quando vídeo **e** áudio têm achado ≥ média | convergência entre fontes independentes |
| 4 | ALERTA se houver achado de severidade alta, ou score ≥ 4 **com convergência**; ATENÇÃO se score ≥ 2; senão NORMAL | achado crítico (ex.: dor no peito) não depende de somar pontos, e várias regras leves de uma só fonte não bastam |

O passo 4 foi corrigido com dado real: a consulta 01 (diarreia, caso benigno) chegava a ALERTA só por
somar quatro termos leves e médios, e passou a ATENÇÃO depois da exigência de convergência.

Parâmetros em `configs/config.yaml` (`fusion:`). Cada evidência cita a modalidade, a regra, a
severidade e, no vídeo, as repetições envolvidas.

Limitações:
- Pesos e limiares provisórios, sem validação clínica: os dados disponíveis não têm rótulo de desfecho.
- Vídeo (reabilitação) e áudio (consulta) vêm de datasets e pessoas diferentes: a convergência é
  demonstrada com um par escolhido para a demo, não com dados de um mesmo paciente.
- O alerta é apoio à triagem e sempre indica avaliação profissional.
