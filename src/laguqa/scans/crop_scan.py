#!/usr/bin/env python3
"""Potong sebagian pindaian buku untuk dibaca lebih teliti.

Membaca notasi angka menuntut ketelitian pada garis balok (beam) di atas angka,
karena garis itulah yang menentukan nilai nada. Pada gambar seukuran halaman
penuh, garis tersebut sering menyatu. Skrip ini memotong satu sistem notasi lalu
memperbesarnya sehingga garis balok terlihat jelas.

Kotak potong dinyatakan dalam proporsi 0..1 terhadap ukuran halaman, sehingga
tidak perlu tahu ukuran piksel gambar sumber.

Pemakaian:
    python scripts/crop_scan.py halaman.jpg --box 0,0.15,1,0.30
    python scripts/crop_scan.py halaman.jpg --rows 6 --row 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps

OUT_DEFAULT = "/tmp/crop"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--box", help="kiri,atas,kanan,bawah dalam proporsi 0..1")
    ap.add_argument("--rows", type=int, help="bagi halaman menjadi n pita mendatar")
    ap.add_argument("--row", type=int, help="ambil pita ke-berapa (mulai dari 1)")
    ap.add_argument("--overlap", type=float, default=0.02, help="tumpang tindih antarpita")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--width", type=int, default=1600, help="lebar hasil setelah perbesaran")
    ap.add_argument("--gray", action="store_true", help="ubah ke abu-abu dan tingkatkan kontras")
    args = ap.parse_args()

    src = Path(args.image)
    if not src.is_file():
        print(f"bukan berkas: {src}", file=sys.stderr)
        return 1

    if args.box:
        l, t, r, b = (float(x) for x in args.box.split(","))
    elif args.rows and args.row:
        h = 1.0 / args.rows
        t = max(0.0, (args.row - 1) * h - args.overlap)
        b = min(1.0, args.row * h + args.overlap)
        l, r = 0.0, 1.0
    else:
        print("beri --box atau pasangan --rows/--row", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode != "RGB":
            im = im.convert("RGB")
        box = (int(l * im.width), int(t * im.height), int(r * im.width), int(b * im.height))
        im = im.crop(box)

        if args.gray:
            im = ImageOps.autocontrast(ImageOps.grayscale(im), cutoff=2).convert("RGB")

        if im.width != args.width:
            ratio = args.width / im.width
            resample = Image.LANCZOS
            im = im.resize((args.width, max(1, int(im.height * ratio))), resample)

        tag = f"r{args.row}" if args.row else f"{l:.2f}-{t:.2f}"
        dst = out / f"{src.stem}__{tag}.jpg"
        im.save(dst, "JPEG", quality=90, optimize=True)
        print(f"{dst}  {im.width}x{im.height}  {dst.stat().st_size // 1024} KB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
