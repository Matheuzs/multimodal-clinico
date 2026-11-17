"""Renderiza o relatório (Markdown e HTML) a partir do dicionário gerado por src.pipeline."""
from __future__ import annotations

from html import escape

SEV_PT = {"low": "baixa", "medium": "média", "high": "alta"}
MOD_PT = {"video": "Vídeo", "audio": "Áudio"}
DISCLAIMER = ("Este relatório é um **apoio à triagem** gerado por métodos automáticos (pose, ASR, regras e dicionário). "
              "Limiares provisórios, sem validação clínica. Não é diagnóstico: requer avaliação profissional.")


def _sev(s):
    return SEV_PT.get(s, s or "-")


def _cell(x):
    return str(x).replace("|", "\\|").replace("\n", " ")


def _table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(_cell(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def render_markdown(r: dict) -> str:
    fz, v, a = r["fusion"], r.get("vision"), r.get("audio")
    L = [f"# Relatório de análise multimodal", f"_Gerado em {r['generated_at']}_", "",
         f"## Resultado: **{fz['level']}** (score {fz['score']:g})", "", fz["message"], ""]
    L += [_table(["Modalidade", "Score"], [["Vídeo", f"{fz['vision_score']:g}" if "video" in fz["modalities"] else "não analisado"],
                                           ["Áudio", f"{fz['audio_score']:g}" if "audio" in fz["modalities"] else "não analisado"],
                                           ["Convergência (bônus)", "sim" if fz["convergence"] else "não"]]), ""]
    L.append("### Evidências")
    if fz["evidences"]:
        L.append(_table(["Origem", "Regra", "Severidade", "Detalhe"],
                        [[MOD_PT[e["modality"]], e["rule"], _sev(e["severity"]),
                          e["message"] + (f" ({e['detail']})" if e["detail"] else "")] for e in fz["evidences"]]))
    else:
        L.append("Nenhuma evidência de desvio.")
    L.append("")
    if v:
        reps = v["repetitions"]
        nflag = sum(1 for x in reps if x["findings"])
        L += ["## Análise de vídeo", f"Arquivo: `{v['file']}` · {v['duration_s']:.0f} s · "
              f"{v['frames_with_pose']}/{v['frames_analyzed']} frames com pose detectada · "
              f"{len(reps)} repetições, {nflag} sinalizadas.", ""]
        if reps:
            L.append(_table(["Rep", "Início–fim (s)", "Pico ombro (°)", "Cotovelo mín. (°)", "Severidade", "Achados"],
                            [[x["index"], f"{x['start_s']}–{x['end_s']}", x["features"].get("peak_shoulder", "-"),
                              x["features"].get("min_elbow", "-"), _sev(x["severity"]),
                              "; ".join(f["rule"] for f in x["findings"]) or "-"] for x in reps]))
        else:
            L.append("Nenhuma repetição segmentada.")
        L.append("")
    if a:
        ac = a["acoustic"]
        L += ["## Análise de áudio", f"Arquivo: `{a['file']}` · {a['duration_s']:.0f} s · transcrição: {a['transcriber']} "
              f"· sentimento: {a['sentiment_analyzer']} · {a['n_words']} palavras", "",
              "### Indicadores acústicos",
              _table(["Métrica", "Valor"], [
                  ["Tempo de fala (s)", ac["speech_s"]], ["Palavras por segundo", ac["speech_rate_wps"]],
                  ["Pausas (total / longas)", f"{ac['n_pauses']} / {ac['n_long_pauses']}"],
                  ["Pausa (% do tempo ativo)", f"{100 * ac['pause_ratio']:.0f}%"],
                  ["Pausas longas por minuto", ac["long_pauses_per_min"]],
                  ["Tendência de energia (dB)", ac["energy_trend_db"]]]), ""]
        L.append("### Termos clínicos detectados")
        pos = [t for t in a["terms"] if not t["negated"]]
        neg = [t for t in a["terms"] if t["negated"]]
        if pos:
            L.append(_table(["Categoria", "Severidade", "Termo", "Frase"],
                            [[t["category"], _sev(t["severity"]), t["term"], t["sentence"][:160]] for t in pos]))
        else:
            L.append("Nenhum termo crítico afirmado.")
        if neg:
            L.append(f"\nTermos negados (ignorados no alerta): {', '.join(sorted({t['category'] for t in neg}))}.")
        if a["negative_ratio"] is not None:
            L.append(f"\nTom negativo: {100 * a['negative_ratio']:.0f}% das {a['n_sentences']} frases.")
        L += ["", "### Trecho da transcrição", "> " + a["transcript"][:600] + ("…" if len(a["transcript"]) > 600 else ""), ""]
    L += ["## Aviso", DISCLAIMER, ""]
    return "\n".join(L)


_CSS = """body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1d2433;line-height:1.5}
table{border-collapse:collapse;width:100%;margin:.5rem 0}th,td{border:1px solid #d5d9e2;padding:.35rem .6rem;text-align:left;font-size:.92rem}
th{background:#f1f3f8}.badge{display:inline-block;padding:.25rem .8rem;border-radius:1rem;color:#fff;font-weight:600}
.NORMAL{background:#2e7d4f}.ATENÇÃO{background:#b7791f}.ALERTA{background:#c0392b}blockquote{border-left:4px solid #d5d9e2;margin:0;padding-left:1rem;color:#444}
.warn{background:#fff8e6;border:1px solid #f0d58a;padding:.7rem 1rem;border-radius:.4rem}"""


def _md_inline(s: str) -> str:
    out, bold = [], False
    for part in escape(s).split("**"):
        out.append(f"<strong>{part}</strong>" if bold else part)
        bold = not bold
    return "".join(out)


def render_html(r: dict) -> str:
    """Converte o Markdown gerado (subconjunto: títulos, tabelas, citação, parágrafos) em HTML autônomo."""
    md = render_markdown(r)
    level = r["fusion"]["level"]
    body, table = [], []

    def flush():
        if table:
            rows = [[c.strip() for c in ln.strip().strip("|").replace("\\|", "\x00").split("|")] for ln in table
                    if not set(ln.replace("|", "").strip()) <= {"-"}]
            head, rest = rows[0], rows[1:]
            body.append("<table><tr>" + "".join(f"<th>{escape(c.replace(chr(0), '|'))}</th>" for c in head) + "</tr>"
                        + "".join("<tr>" + "".join(f"<td>{escape(c.replace(chr(0), '|'))}</td>" for c in row) + "</tr>"
                                  for row in rest) + "</table>")
            table.clear()

    for ln in md.splitlines():
        if ln.startswith("|"):
            table.append(ln)
            continue
        flush()
        if ln.startswith("### "):
            body.append(f"<h3>{escape(ln[4:])}</h3>")
        elif ln.startswith("## Resultado"):
            body.append(f"<h2>Resultado: <span class='badge {level}'>{level}</span> "
                        f"<small>score {r['fusion']['score']:g}</small></h2>")
        elif ln.startswith("## Aviso"):
            body.append("<h2>Aviso</h2>")
        elif ln.startswith("## "):
            body.append(f"<h2>{escape(ln[3:])}</h2>")
        elif ln.startswith("# "):
            body.append(f"<h1>{escape(ln[2:])}</h1>")
        elif ln.startswith("> "):
            body.append(f"<blockquote>{escape(ln[2:])}</blockquote>")
        elif ln.strip():
            cls = " class='warn'" if ln == DISCLAIMER else ""
            body.append(f"<p{cls}>{_md_inline(ln.replace('`', '').strip('_'))}</p>")
    flush()
    return ("<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><title>Relatório multimodal</title>"
            f"<style>{_CSS}</style></head><body>" + "\n".join(body) + "</body></html>")
