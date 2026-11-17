# Resultados da transcrição de áudio (rascunho para o relatório técnico)

Dataset: PriMock57 (consultas simuladas; áudio do paciente em arquivo separado do médico), 3 consultas
(day1_consultation01..03, 1584 palavras de referência). Referência: transcrição humana (TextGrid).
WER com normalização: minúsculas, sem pontuação, sem interjeições (um, uh, hmm...).
Reprodução: `python -m scripts.eval_asr --provider local_whisper --model small --n 3`

| Modelo Whisper | WER consulta 01 | 02 | 03 | WER médio ponderado | tempo por consulta |
|---|---|---|---|---|---|
| base (139 MB) | 19,6% | 23,8% | 31,9% | 25,1% | 9–13 s |
| small (461 MB) | 13,7% | 19,2% | 19,1% | 17,7% | 21–29 s |

Erros do `base`: termos médicos trocados (itching->teaching, eczema->exam, diarrhea->diaries,
stool->stall) e perda de falas curtas ("yeah", "I", "like"). Só 3 trechos de 5+ palavras foram perdidos.

## Limitações
- Apenas 3 consultas do primeiro dia de gravação; um único locutor por consulta.
- Fala simulada (atores) em inglês britânico, não pacientes reais.
- Números e variações de grafia ("OK"/"okay", "six"/"6") podem contar como erro.

## Termos clínicos nas 3 consultas
Contagem de ocorrências (referência humana) reconhecidas no texto do Whisper, para a lista fixa
itching, itchy, eczema, diarrhea, stool, pain, blood, fever, rash, cough (8 aparecem nas consultas):
- base: 6/19 (itching 0/3, eczema 0/4, diarrhea 0/2, stool 0/2)
- small: 19/19
Amostra pequena (19 ocorrências) e lista escolhida após ver os erros do `base`: indica a tendência, não é um benchmark.
Por isso `whisper_model: small` é o padrão em `configs/config.yaml`.
