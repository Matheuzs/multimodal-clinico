"""Extrai só alguns vídeos do videos.zip (2,7 GB) do REHAB24-6 no Zenodo,
usando requisições HTTP Range (remotezip). Sem Range, cai no download completo.

  python scripts/fetch_rehab24.py --list
  python scripts/fetch_rehab24.py --pattern PM_001-Camera17 --max 1 --out data/raw/rehab24/videos
"""
import argparse
import re
import sys
from pathlib import Path

URL = "https://zenodo.org/records/13305826/files/videos.zip?download=1"
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--pattern", default="PM_001-Camera17", help="regex aplicado ao nome do arquivo")
    ap.add_argument("--max", type=int, default=1)
    ap.add_argument("--out", default="data/raw/rehab24/videos")
    a = ap.parse_args()

    try:
        from remotezip import RemoteZip
    except ImportError:
        print("Instale: pip install remotezip", file=sys.stderr)
        return 1

    with RemoteZip(URL) as z:
        infos = [i for i in z.infolist() if i.filename.lower().endswith(VIDEO_EXTS)]
        if a.list:
            for i in infos:
                print(f"{i.file_size/1e6:8.1f} MB  {i.filename}")
            print(f"{len(infos)} vídeos")
            return 0
        rx = re.compile(a.pattern)
        chosen = [i for i in infos if rx.search(i.filename)][: a.max]
        if not chosen:
            print("Nenhum vídeo casou com o padrão; rode --list.", file=sys.stderr)
            return 2
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        for i in chosen:
            dest = out / Path(i.filename).name
            if dest.exists() and dest.stat().st_size == i.file_size:
                print("ok  ", dest)
                continue
            print("get ", dest, f"({i.file_size/1e6:.1f} MB)")
            with z.open(i) as src, open(dest, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
