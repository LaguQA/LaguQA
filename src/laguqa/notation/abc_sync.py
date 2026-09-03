#!/usr/bin/env python3
"""Satukan berkas ABC per lagu ke dalam dataset.

Sumber kebenaran notasi adalah berkas `data/abc/<id>_<judul>.abc`, bukan sel di
dalam spreadsheet. Alasannya, notasi ABC berisi banyak baris dan tanda kutip
sehingga sulit dibaca maupun dibandingkan ketika disimpan sebagai satu sel.
Berkas terpisah bisa dibuka, disunting, dan diperiksa validator satu per satu.

Skrip ini membaca seluruh berkas di `data/abc/`, menjalankan validator pada
masing-masing, lalu menulis dua kolom ke dataset:

    abc_notation  isi berkas apa adanya
    abc_status    terverifikasi | mentah | kosong

"terverifikasi" berarti validator tidak menemukan pelanggaran. "mentah" berarti
notasi sudah ada tetapi masih menyisakan temuan, jadi belum boleh dipakai
sebagai kunci jawaban soal.

Mode bawaan dry-run. Tambahkan --apply untuk menulis.

Pemakaian:
    python scripts/07_sync_abc.py
    python scripts/07_sync_abc.py --apply
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

from laguqa.notation.abc_validate import report
from laguqa.paths import ABC_DIR, CSV_PATH

csv.field_size_limit(10**8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not CSV_PATH.exists():
        print(f"tidak ditemukan: {CSV_PATH}", file=sys.stderr)
        return 1
    ABC_DIR.mkdir(parents=True, exist_ok=True)

    with open(CSV_PATH, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys())
    if "abc_status" not in fields:
        fields.insert(fields.index("abc_notation") + 1, "abc_status")

    by_id: dict[str, Path] = {}
    for p in sorted(ABC_DIR.glob("*.abc")):
        m = re.match(r"(\d+)_", p.name)
        if not m:
            print(f"  LEWATI (tanpa awalan nomor): {p.name}")
            continue
        rid = str(int(m.group(1)))
        if rid in by_id:
            print(f"  BENTROK id {rid}: {by_id[rid].name} dan {p.name}")
            return 1
        by_id[rid] = p

    print(f"{'MODE APPLY' if args.apply else 'MODE DRY-RUN (tidak ada yang ditulis)'}")
    print(f"berkas abc : {len(by_id)}")
    print(f"baris data : {len(rows)}\n")

    tally = {"terverifikasi": 0, "mentah": 0, "kosong": 0}
    print(f"{'id':>4}  {'judul':<30} {'status':<14} temuan  peringatan")
    for r in rows:
        rid = (r.get("id") or "").strip()
        path = by_id.get(rid)
        if path is None:
            r["abc_notation"] = ""
            r["abc_status"] = "kosong"
            tally["kosong"] += 1
            continue

        text = path.read_text(encoding="utf-8")
        buf = io.StringIO()
        with redirect_stdout(buf):
            findings, stats = report(path.name, text, quiet=True)
        status = "terverifikasi" if findings == 0 else "mentah"
        r["abc_notation"] = text
        r["abc_status"] = status
        tally[status] += 1
        print(
            f"{rid:>4}  {(r.get('title') or '')[:30]:<30} {status:<14} "
            f"{findings:>6}  {stats.get('warnings', 0):>10}"
        )

    yatim = sorted(set(by_id) - {(r.get("id") or "").strip() for r in rows})
    if yatim:
        print(f"\nberkas abc tanpa baris dataset: {yatim}")

    print(f"\nterverifikasi : {tally['terverifikasi']}")
    print(f"mentah        : {tally['mentah']}")
    print(f"kosong        : {tally['kosong']}")

    if args.apply:
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"\nditulis: {CSV_PATH}")
    else:
        print("\nJalankan dengan --apply untuk menulis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
