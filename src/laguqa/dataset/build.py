#!/usr/bin/env python3
"""Pembersih integritas dataset LaguQA.

Menormalkan nama berkas pindaian, memperbaiki bentrok penomoran, mengisi kolom
image_filename, dan melaporkan sisa masalah yang butuh keputusan manusia.

Mode bawaan adalah dry-run: tidak ada berkas yang diubah. Tambahkan --apply
untuk benar-benar menjalankan perubahan.

Pemakaian:
    python scripts/03_build_dataset.py             # laporan saja
    python scripts/03_build_dataset.py --apply     # jalankan perubahan
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

from laguqa.paths import CSV_PATH as OUT_CSV, RAW_SCANS_DIR as IMAGES, XLSX_PATH as XLSX

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

# Perbaikan yang sudah dipastikan dari audit
TITLE_FIXES = {
    "19": "Bubuy Bulan",
    "76": "Bhinneka Tunggal Ika",
}

# Berkas yang nomornya bentrok dan harus dipindahkan ke nomor lain
RENUMBER = {
    "102_juwita_malam.jpg": "103_juwita_malam.jpg",
}

# Berkas tanpa awalan nomor yang menunggu identifikasi manual
UNNUMBERED_HINTS = {
    "pemilu_2.jpg": "kemungkinan halaman 2 dari id 87 (Pemilu)",
}


def slugify(name: str) -> str:
    """Normalkan nama berkas: huruf kecil, garis bawah, tanpa diakritik."""
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, ""
    stem = unicodedata.normalize("NFKD", stem)
    stem = "".join(c for c in stem if not unicodedata.combining(c))
    stem = stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return f"{stem}.{ext.lower()}" if ext else stem


def read_xlsx(path: Path) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as zf:
        strings = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{NS}si"):
                strings.append("".join(t.text or "" for t in si.iter(f"{NS}t")))
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    rows = []
    for row in sheet.iter(f"{NS}row"):
        cells: dict[str, str] = {}
        for c in row.findall(f"{NS}c"):
            ref = c.get("r") or ""
            col = re.match(r"[A-Z]+", ref)
            if not col:
                continue
            ctype = c.get("t")
            v = c.find(f"{NS}v")
            is_ = c.find(f"{NS}is")
            if ctype == "s" and v is not None:
                val = strings[int(v.text or 0)]
            elif ctype == "inlineStr" and is_ is not None:
                val = "".join(t.text or "" for t in is_.iter(f"{NS}t"))
            else:
                val = (v.text or "") if v is not None else ""
            cells[col.group()] = val
        rows.append(cells)

    if not rows:
        return []
    header = rows[0]
    cols = sorted(header, key=lambda c: (len(c), c))
    names = [header[c] for c in cols]
    return [{names[i]: r.get(c, "") for i, c in enumerate(cols)} for r in rows[1:]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="benar-benar jalankan perubahan")
    args = ap.parse_args()
    apply = args.apply

    if not XLSX.exists():
        print(f"tidak ditemukan: {XLSX}", file=sys.stderr)
        return 1
    if not IMAGES.is_dir():
        print(f"tidak ditemukan: {IMAGES}", file=sys.stderr)
        return 1

    recs = read_xlsx(XLSX)
    files = sorted(p.name for p in IMAGES.iterdir() if p.is_file())

    print(f"{'MODE APPLY' if apply else 'MODE DRY-RUN (tidak ada yang diubah)'}")
    print(f"baris dataset : {len(recs)}")
    print(f"berkas gambar : {len(files)}")

    # --- 1. rencana penamaan ulang -----------------------------------------
    print("\n--- 1. Penamaan ulang berkas ---")
    renames: dict[str, str] = {}
    for f in files:
        target = RENUMBER.get(f, f)
        target = slugify(target)
        if target != f:
            renames[f] = target

    collisions = defaultdict(list)
    for src, dst in renames.items():
        collisions[dst].append(src)
    for dst, srcs in collisions.items():
        if len(srcs) > 1 or (dst in files and dst not in renames):
            print(f"  BENTROK  {dst} <- {srcs}")

    for src, dst in sorted(renames.items()):
        tag = "RENUMBER" if src in RENUMBER else "normalkan"
        print(f"  {tag:9s} {src}  ->  {dst}")
    if not renames:
        print("  tidak ada yang perlu diubah")

    if apply:
        for src, dst in renames.items():
            s, d = IMAGES / src, IMAGES / dst
            if d.exists() and s.resolve() != d.resolve():
                print(f"  LEWATI (tujuan sudah ada): {dst}")
                continue
            shutil.move(str(s), str(d))
        files = sorted(p.name for p in IMAGES.iterdir() if p.is_file())

    final_names = {renames.get(f, f) for f in files}

    # --- 2. pemetaan nomor -> berkas ---------------------------------------
    by_num: dict[int, list[str]] = defaultdict(list)
    unnumbered: list[str] = []
    for f in sorted(final_names):
        m = re.match(r"(\d+)_", f)
        if m:
            by_num[int(m.group(1))].append(f)
        else:
            unnumbered.append(f)

    print("\n--- 2. Berkas tanpa awalan nomor ---")
    for f in unnumbered:
        hint = UNNUMBERED_HINTS.get(f, "belum teridentifikasi")
        print(f"  {f:34s} {hint}")
    print(f"  total: {len(unnumbered)} berkas menunggu identifikasi manual")

    # --- 3. isi image_filename dan perbaiki judul --------------------------
    print("\n--- 3. Perbaikan baris dataset ---")
    fixed_title = changed = broken = 0
    for r in recs:
        rid = (r.get("id") or "").strip()
        if rid in TITLE_FIXES and r.get("title") != TITLE_FIXES[rid]:
            print(f"  judul   id {rid:>3}: '{r.get('title')}' -> '{TITLE_FIXES[rid]}'")
            r["title"] = TITLE_FIXES[rid]
            fixed_title += 1

        current = (r.get("image_filename") or "").strip()
        # Bangun ulang dari disk, bukan sekadar isi kalau kosong. Nama berkas di
        # disk adalah sumber kebenaran, sehingga entri ganda maupun halaman yang
        # terlewat pada data lama ikut terkoreksi.
        matched = sorted(by_num.get(int(rid), [])) if rid.isdigit() else []
        rebuilt = ",".join(matched)

        if current and current != rebuilt:
            was = {slugify(x.strip()) for x in current.split(",") if x.strip()}
            now = set(matched)
            hilang = sorted(was - now)
            tambah = sorted(now - was)
            if hilang:
                broken += 1
                print(f"  RUSAK   id {rid:>3}: {hilang} tercatat tapi tidak ada di disk")
            if tambah:
                print(f"  TAMBAH  id {rid:>3}: {tambah} ada di disk tapi belum tercatat")
            if not hilang and not tambah:
                print(f"  RAPIKAN id {rid:>3}: penulisan atau urutan nama dirapikan")
            changed += 1

        r["image_filename"] = rebuilt

    print(f"  judul diperbaiki        : {fixed_title}")
    print(f"  image_filename diubah   : {changed}")
    print(f"  referensi rusak         : {broken}")

    # berkas di disk yang tidak dirujuk baris mana pun
    dirujuk = {f for r in recs for f in (r.get("image_filename") or "").split(",") if f}
    yatim = sorted(final_names - dirujuk)
    print(f"  berkas tidak dirujuk    : {len(yatim)}")
    for f in yatim:
        print(f"        {f}")

    # --- 4. sisa masalah ----------------------------------------------------
    print("\n--- 4. Butuh keputusan manusia ---")
    no_image = [r for r in recs if not (r.get("image_filename") or "").strip()]
    for r in no_image:
        print(f"  id {r.get('id'):>3} tanpa gambar : {r.get('title')}")

    titles: dict[str, list[str]] = defaultdict(list)
    for r in recs:
        titles[(r.get("title") or "").strip().lower()].append(str(r.get("id")))
    for t, ids in titles.items():
        if len(ids) > 1:
            print(f"  judul duplikat        : '{t}' pada id {ids}")

    for r in recs:
        if re.search(r"\s\d+$", (r.get("title") or "").strip()):
            print(f"  judul sementara       : id {r.get('id')} '{r.get('title')}'")

    # --- 5. keluaran --------------------------------------------------------
    # Skrip ini membangun ulang dari dataset.xlsx, sedangkan pekerjaan tahap
    # berikutnya menulis ke data/laguqa.csv. Kolom yang hanya ada di CSV,
    # misalnya abc_status hasil abc_sync.py, karena itu harus dipertahankan.
    # Tanpa penjagaan ini, menjalankan ulang skrip akan menghapus transkripsi
    # yang sudah lolos verifikasi tanpa peringatan apa pun.
    fields = list(recs[0].keys())
    if OUT_CSV.exists():
        with open(OUT_CSV, encoding="utf-8") as fh:
            existing = {(r.get("id") or "").strip(): r for r in csv.DictReader(fh)}
        extra = [c for c in (next(iter(existing.values()), {}) or {}) if c not in fields]
        if extra:
            print(f"\n--- 5. Kolom yang dipertahankan dari {OUT_CSV.name} ---")
            for c in extra:
                terisi = sum(1 for r in existing.values() if (r.get(c) or "").strip())
                print(f"  {c:16s} {terisi} baris terisi")
            fields += extra
            for r in recs:
                prev = existing.get((r.get("id") or "").strip(), {})
                for c in extra:
                    r[c] = prev.get(c, "")

    if apply:
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(recs)
        print(f"\nditulis: {OUT_CSV}")
    else:
        print(f"\nJalankan dengan --apply untuk menerapkan dan menulis {OUT_CSV.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
