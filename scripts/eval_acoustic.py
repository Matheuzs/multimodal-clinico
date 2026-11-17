"""Métricas acústicas de paciente e médico nas consultas baixadas do PriMock57.

  python -m scripts.eval_acoustic
"""
from src.audio.acoustic import extract_features
from src.audio.textgrid import reference_text
from src.audio.metrics import normalize
from src.config import ROOT
b=ROOT/"data/raw/primock57"
print("quem  cons  fala_s ativo_s  pausas  longas  pausa%  maxP  longas/min  wps   dB_med  tendência_dB")
for who,other in (("patient","doctor"),("doctor","patient")):
    for n in ("01","02","03","04","05"):
        stem=f"day1_consultation{n}"
        nw=len(normalize(reference_text(b/f"transcripts/{stem}_{who}.TextGrid")))
        f=extract_features(b/f"audio/{stem}_{who}.wav",nw,b/f"audio/{stem}_{other}.wav")
        print(f"{who[:3]}  {n}  {f.speech_s:6.0f} {f.active_s:7.0f} {f.n_pauses:7d} {f.n_long_pauses:7d} {100*f.pause_ratio:6.0f} {f.max_pause_s:6.1f} {f.long_pauses_per_min:9.2f} {f.speech_rate_wps:5.2f} {f.rms_db_mean:7.1f} {f.energy_trend_db if f.energy_trend_db is None else round(f.energy_trend_db,1)}")
