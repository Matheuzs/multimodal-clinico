#!/usr/bin/env bash
# Baixa os datasets do MVP (idempotente: pula o que já existe).
#   ./scripts/download_data.sh primock          # áudio (PriMock57)
#   ./scripts/download_data.sh rehab-list       # lista os arquivos do videos.zip
#   ./scripts/download_data.sh rehab            # vídeos (REHAB24-6)
#   ./scripts/download_data.sh all
# Variáveis: N_CONSULTATIONS (padrão 5), REHAB_PATTERN (regex, padrão "PM_001-Camera17"), REHAB_MAX (padrão 1)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"
N_CONSULTATIONS="${N_CONSULTATIONS:-5}"
PRIMOCK_MEDIA="https://media.githubusercontent.com/media/babylonhealth/primock57/main"
PRIMOCK_RAW="https://raw.githubusercontent.com/babylonhealth/primock57/main"
ZENODO="https://zenodo.org/records/13305826/files"

fetch() {  # fetch URL DEST
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then echo "ok   $dest"; return; fi
  mkdir -p "$(dirname "$dest")"
  echo "get  $dest"
  curl -fL --retry 3 --retry-delay 2 -sS -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

primock() {
  local out="$RAW/primock57" i id
  for i in $(seq 1 "$N_CONSULTATIONS"); do
    id="day1_consultation$(printf '%02d' "$i")"
    for who in doctor patient; do
      fetch "$PRIMOCK_MEDIA/audio/${id}_${who}.wav" "$out/audio/${id}_${who}.wav"
      fetch "$PRIMOCK_RAW/transcripts/${id}_${who}.TextGrid" "$out/transcripts/${id}_${who}.TextGrid"
    done
    fetch "$PRIMOCK_RAW/notes/${id}.json" "$out/notes/${id}.json"
  done
  fetch "$PRIMOCK_RAW/LICENSE.md" "$out/LICENSE.md"
}

rehab_meta() {
  local out="$RAW/rehab24"
  for f in Segmentation.csv Segmentation.txt joints_names.txt; do
    fetch "$ZENODO/$f?download=1" "$out/$f"
  done
}

rehab_list() { python3 "$ROOT/scripts/fetch_rehab24.py" --list; }
rehab() {
  rehab_meta
  python3 "$ROOT/scripts/fetch_rehab24.py" --pattern "${REHAB_PATTERN:-PM_001-Camera17}" --max "${REHAB_MAX:-1}" \
    --out "$RAW/rehab24/videos"
}

case "${1:-all}" in
  primock)    primock ;;
  rehab-list) rehab_list ;;
  rehab)      rehab ;;
  all)        primock; rehab ;;
  *) echo "uso: $0 {primock|rehab-list|rehab|all}"; exit 1 ;;
esac
du -sh "$RAW" 2>/dev/null || true
