#!/usr/bin/env python3
"""Ratakan pencahayaan pindaian buku sebelum dikirim ke model vision.

Pindaian dibuat dengan kamera ponsel, sehingga separuh halaman sering lebih
gelap daripada separuh lainnya. Latar halaman jadi kelabu, bukan putih, dan
garis balok tipis di atas angka nyaris menyatu dengan kertas. Padahal garis
balok itulah yang menentukan nilai nada, sehingga bagian inilah yang paling
mahal kalau terbaca salah.

Perataan dikerjakan dengan koreksi medan datar: citra dibagi oleh versi dirinya
sendiri yang diburamkan sangat kuat. Versi buram itu berisi pola pencahayaan
tanpa detail tulisan, jadi pembagian tersebut membuang gradien bayangan dan
menyisakan tulisannya. Cara ini bekerja pada bayangan miring, yang tidak bisa
diperbaiki oleh autocontrast biasa karena autocontrast berlaku global.

Pemakaian:
    python scripts/02_prep_scans.py                        # seluruh halaman mentah
    python -m laguqa.scans.prep_scan --out /tmp/uji --stats berkas.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageStat

# Beberapa halaman difoto dengan buku diputar, dan metadata EXIF tidak
# merekamnya sehingga exif_transpose tidak menolong. Nilainya derajat searah
# jarum jam yang perlu diterapkan supaya halaman kembali tegak.
ROTATE = {
    "42_ma_rencong.jpg": 90,
    "45_o_ina_ni_keke.jpg": 90,
    "77_hymne_kemerdekaan_p2.jpg": 90,
}


def flatten(im: Image.Image, blur: float, cutoff: float, sharpen: bool) -> Image.Image:
    """Buang gradien pencahayaan, lalu regangkan kontras."""
    g = ImageOps.grayscale(im)

    # Radius buram harus jauh lebih besar daripada tebal huruf, supaya yang
    # tersisa hanya pola cahaya. Diskalakan terhadap lebar citra agar hasilnya
    # setara pada resolusi yang berbeda-beda.
    radius = max(8, int(g.width * blur / 1000))
    background = g.filter(ImageFilter.GaussianBlur(radius))

    a = np.asarray(g, dtype=np.float32) + 1.0
    b = np.asarray(background, dtype=np.float32) + 1.0
    flat = Image.fromarray(np.clip(255.0 * a / b, 0, 255).astype(np.uint8), mode="L")

    flat = ImageOps.autocontrast(flat, cutoff=cutoff)
    if sharpen:
        flat = flat.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=3))
    return flat


def stats(im: Image.Image) -> tuple[float, float]:
    s = ImageStat.Stat(ImageOps.grayscale(im))
    return s.mean[0], s.stddev[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blur", type=float, default=25, help="radius buram per 1000 piksel lebar")
    ap.add_argument("--cutoff", type=float, default=1.0)
    ap.add_argument("--no-sharpen", action="store_true")
    ap.add_argument("--width", type=int, default=0, help="0 berarti ukuran asli")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--stats", action="store_true", help="tampilkan rerata dan kontras")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0

    for path in args.files:
        src = Path(path)
        if not src.is_file():
            print(f"lewati (bukan berkas): {src}", file=sys.stderr)
            continue

        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im)
            deg = ROTATE.get(src.name, 0)
            if deg:
                # rotate() berlawanan jarum jam, jadi sudutnya dibalik
                im = im.rotate(-deg, expand=True)
            before = stats(im) if args.stats else None
            flat = flatten(im, args.blur, args.cutoff, not args.no_sharpen)

            if args.width and flat.width > args.width:
                ratio = args.width / flat.width
                flat = flat.resize((args.width, int(flat.height * ratio)), Image.LANCZOS)

            dst = out / f"{src.stem}.jpg"
            flat.convert("RGB").save(dst, "JPEG", quality=args.quality, optimize=True)
            n += 1

            if args.stats:
                after = stats(flat)
                print(
                    f"{src.name:44s} rerata {before[0]:5.1f} -> {after[0]:5.1f}   "
                    f"kontras {before[1]:5.1f} -> {after[1]:5.1f}"
                )
            else:
                print(f"{src.name:44s} -> {dst.name}")

    print(f"\n{n} berkas diproses ke {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
