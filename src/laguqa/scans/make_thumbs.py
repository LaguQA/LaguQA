#!/usr/bin/env python3
"""Buat gambar kecil dari pindaian buku untuk keperluan identifikasi judul.

Pindaian asli berukuran sekitar 5 MB dan 4624x3472 piksel, terlalu besar untuk
dibaca berulang kali. Skrip ini membuat versi kecil, dengan opsi memotong bagian
atas halaman tempat judul biasanya dicetak.

Pemakaian:
    python scripts/make_thumbs.py --out /tmp/thumbs img1.jpg img2.jpg
    python scripts/make_thumbs.py --out /tmp/thumbs --top 0.35 *.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True, help="direktori keluaran")
    ap.add_argument("--width", type=int, default=1400, help="lebar maksimum hasil")
    ap.add_argument(
        "--top",
        type=float,
        default=0.0,
        help="ambil hanya bagian atas sebesar proporsi ini (0 = halaman penuh)",
    )
    ap.add_argument("--quality", type=int, default=82)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for path in args.files:
        src = Path(path)
        if not src.is_file():
            print(f"lewati (bukan berkas): {src}", file=sys.stderr)
            continue

        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)  # hormati orientasi dari metadata kamera
            if im.mode != "RGB":
                im = im.convert("RGB")

            if args.top > 0:
                h = int(im.height * args.top)
                im = im.crop((0, 0, im.width, h))

            if im.width > args.width:
                ratio = args.width / im.width
                im = im.resize((args.width, int(im.height * ratio)), Image.LANCZOS)

            dst = out / f"{src.stem}.jpg"
            im.save(dst, "JPEG", quality=args.quality, optimize=True)
            kb = dst.stat().st_size // 1024
            print(f"{src.name:34s} -> {dst.name:34s} {im.width}x{im.height}  {kb} KB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
