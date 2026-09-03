#!/usr/bin/env python3
"""Siapkan berkas .abc kosong untuk seluruh lagu, satu berkas satu lagu.

Keluaran model ditempel manual dari jendela obrolan, satu lagu satu percakapan.
Mengetik nama berkasnya sendiri setiap kali membuka peluang salah ketik, dan
nama berkas itulah satu-satunya sumber nomor lagu bagi abc_ingest.py. Sekali
salah, keluaran tersimpan di nomor lagu yang keliru. Karena itu seluruh nama
dibuat sekaligus di muka dari dataset, dan pengisian tinggal menempel isinya.

Nama berkas dibentuk memakai slugify() milik abc_ingest, bukan salinannya,
supaya nama yang dibuat di sini dan nama yang diharapkan di sana tidak mungkin
berbeda.

Pemakaian:
    python scripts/04_make_abc_stubs.py              # laporan kemajuan
    python scripts/04_make_abc_stubs.py --apply      # buat yang belum ada
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from laguqa.notation.abc_ingest import STUB_MARK as MARK, scans_for, slugify
from laguqa.paths import ABC_RAW_DIR, CSV_PATH


def is_stub(text: str) -> bool:
    return MARK in text.splitlines()[0] if text.strip() else True


def stub_text(rid: str, title: str, berkas: list[str]) -> str:
    return "\n".join([
        f"{MARK} - ganti SELURUH isi berkas ini dengan keluaran model",
        f"% lagu {rid} - {title}",
        f"% gambar: {', '.join(berkas) if berkas else 'TIDAK ADA'}",
        "",
    ])


def rows() -> list[dict]:
    csv.field_size_limit(10**8)
    with open(CSV_PATH, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None,
                    help=f"tempat berkas disiapkan (bawaan: {ABC_RAW_DIR})")
    ap.add_argument("--apply", action="store_true", help="tulis perubahan")
    args = ap.parse_args()

    out = Path(args.dir) if args.dir else ABC_RAW_DIR
    out.mkdir(parents=True, exist_ok=True)

    dibuat: list[str] = []
    diganti: list[str] = []
    terisi: list[str] = []
    kosong: list[str] = []
    bingung: list[str] = []

    for r in rows():
        rid = str(int(r["id"]))
        title = (r["title"] or "").strip()
        nama = f"{int(rid):03d}_{slugify(title)}.abc"
        dst = out / nama
        berkas = scans_for(rid)

        ada = dst
        if not dst.exists():
            # Berkas dengan nomor yang sama tetapi nama lain berarti sudah
            # dikerjakan dengan ejaan berbeda. Isinya pekerjaan sungguhan, jadi
            # namanya yang diseragamkan, bukan berkasnya yang ditimpa.
            lain = [p for p in out.glob(f"{int(rid):03d}_*.abc") if p != dst]
            if len(lain) == 1:
                diganti.append(f"{lain[0].name} -> {nama}")
                if args.apply:
                    lain[0].rename(dst)
                else:
                    ada = lain[0]
            elif len(lain) > 1:
                bingung.append(f"{rid}: {', '.join(p.name for p in lain)}")
                continue
            else:
                dibuat.append(nama)
                if args.apply:
                    dst.write_text(stub_text(rid, title, berkas), encoding="utf-8")
                kosong.append(nama)
                continue

        if is_stub(ada.read_text(encoding="utf-8")):
            kosong.append(nama)
        else:
            terisi.append(nama)

    for nama in diganti:
        print(f"  ganti nama  {nama}")
    for nama in bingung:
        print(f"  [BINGUNG] lebih dari satu berkas bernomor sama, {nama}")

    total = len(terisi) + len(kosong)
    print(f"\n{out}: {len(terisi)}/{total} terisi, {len(kosong)} kosong")
    if dibuat:
        kata = "dibuat" if args.apply else "akan dibuat"
        print(f"{len(dibuat)} berkas kosong {kata}")
    if kosong:
        print("berikutnya: " + ", ".join(kosong[:5]) + ("..." if len(kosong) > 5 else ""))
    if not args.apply and (dibuat or diganti):
        print("\nJalankan dengan --apply untuk menulis")
    return 0


if __name__ == "__main__":
    sys.exit(main())
