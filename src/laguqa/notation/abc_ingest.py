#!/usr/bin/env python3
"""Terima keluaran mentah model, periksa kontraknya, simpan ke data/abc/.

Keluaran model datang sebagai teks tempelan dari jendela obrolan, biasanya
terbungkus pagar markdown. Skrip ini membuka bungkusnya, memastikan kepala
berkasnya memenuhi kontrak pada prompts/prompt.txt, mengisi sendiri nomor lagu
dan daftar gambar dari nama berkas masukan, lalu menjalankan validator.

Kontrak kepala berkas dipakai supaya audit bisa dikerjakan mesin. Nomor lagu
dan nama gambar sengaja TIDAK diminta dari model, karena model tidak melihat
nama berkas gambar yang diunggah sehingga jawabannya pasti karangan.

Beri nama berkas masukan dengan nomor lagunya, misalnya masuk/006.txt.

Pemakaian:
    python scripts/05_ingest_abc.py keluaran.txt
    pbpaste | python scripts/05_ingest_abc.py -
    python scripts/05_ingest_abc.py ../sumber/masuk/*.txt --apply
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from laguqa.notation.abc_validate import report
from laguqa.paths import ABC_DIR, CSV_PATH, SCANS_DIR

FENCE_RE = re.compile(r"```[a-zA-Z]*\s*\n(.*?)```", re.S)

# Hanya dua yang diminta dari model, yaitu yang memang terbaca di gambar.
# laguqa-id dan laguqa-berkas justru TIDAK boleh diminta: model tidak melihat
# nama berkas gambar yang diunggah, sehingga jawabannya pasti karangan. Sekali
# percobaan menghasilkan "id 11" untuk Bungong Jeumpa yang sebenarnya id 0,
# diambil dari nomor halaman. Keduanya diisi skrip ini dari nama berkas masukan.
DIRECTIVES = ["laguqa-do", "laguqa-halaman"]
FIELDS = ["X", "T", "C", "O", "Q", "M", "L", "K"]

# Penanda berkas kosong yang disiapkan abc_stub.py. Berkas begini dilewati
# tanpa keluhan, supaya seluruh direktori bisa diperiksa sekaligus sejak lagu
# pertama dikerjakan, bukan menunggu ke-107 lagu selesai.
STUB_MARK = "% BELUM DIISI"

# Kemiripan judul minimum sebelum berkas dianggap tersimpan di nomor yang
# salah. Gunanya menangkap keluaran lagu lain, bukan menyamakan ejaan, jadi
# ambangnya longgar: "bhineka tunggal ika" lawan "bhinneka tunggal ika"
# bernilai 0.97, sedangkan dua judul lagu berbeda jatuh jauh di bawahnya.
MIRIP_MIN = 0.85


def unfence(text: str) -> str:
    """Ambil isi pagar markdown kalau ada, kalau tidak kembalikan apa adanya."""
    blocks = FENCE_RE.findall(text)
    if blocks:
        # Ambil blok yang mengandung %abc, supaya contoh atau basa-basi tidak ikut.
        for b in blocks:
            if "%abc" in b:
                return b.strip("\n")
        return max(blocks, key=len).strip("\n")
    return text.strip("\n")


def slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def judul_mirip(a: str, b: str) -> float:
    """Nilai kemiripan dua judul, 1.0 kalau sama persis.

    Angka di dalam judul diperlakukan sebagai pembeda mutlak. Dataset memuat
    "Tanah Airku 1" dan "Tanah Airku 2", dua lagu berlainan yang hanya
    dibedakan angkanya; kemiripan hurufnya 0.92, cukup tinggi untuk lolos
    ambang yang dipasang longgar demi menoleransi ejaan. Selisih angka karena
    itu langsung bernilai nol, tidak ikut ditimbang.
    """
    sa, sb = slugify(a), slugify(b)
    if re.findall(r"\d+", sa) != re.findall(r"\d+", sb):
        return 0.0
    return SequenceMatcher(None, sa, sb).ratio()


def check_contract(text: str) -> tuple[dict, list[str]]:
    """Kembalikan (nilai terbaca, daftar pelanggaran kontrak)."""
    lines = [l.rstrip() for l in text.splitlines()]
    bad: list[str] = []
    got: dict[str, str] = {}

    if not lines or lines[0].strip() != "%abc-2.1":
        bad.append("baris pertama bukan %abc-2.1")

    for name in DIRECTIVES:
        # Komentar biasa "% nama nilai". Bentuk "%%nama" sempat dipakai tetapi
        # ditinggalkan: di ABC 2.1 "%%" adalah stylesheet directive, sehingga
        # parser mengeluh "Unknown directive" untuk nama yang tidak dikenalnya.
        # Bentuk lama tetap diterima supaya keluaran yang terlanjur dibuat
        # tidak perlu diulang.
        m = re.search(rf"^%%?[ \t]*{re.escape(name)}[ \t]+(.*)$", text, re.M)
        if not m:
            bad.append(f"tidak ada baris % {name}")
        else:
            got[name] = m.group(1).strip()

    seen: list[str] = []
    for tag in FIELDS:
        m = re.search(rf"^{tag}:(.*)$", text, re.M)
        if not m:
            bad.append(f"tidak ada field {tag}:")
        else:
            got[tag] = m.group(1).strip()
            seen.append(tag)
            if not got[tag]:
                bad.append(f"field {tag}: kosong, seharusnya diisi '-' kalau tidak ada")

    if seen != [t for t in FIELDS if t in seen]:
        bad.append(f"urutan field menyimpang: {seen}")

    if got.get("L") and got["L"] != "1/8":
        bad.append(f"L: harus 1/8, ditemukan {got['L']}")

    if re.search(r"^%%?[ \t]*laguqa-(id|berkas)\b", text, re.M):
        bad.append(
            "keluaran memuat laguqa-id atau laguqa-berkas; keduanya diisi skrip, "
            "bukan model, karena model tidak melihat nama berkas gambar"
        )

    # Validator sengaja tidak memeriksa bar pertama dan terakhir, supaya bar
    # gantung tidak dilaporkan salah. Akibatnya berkas yang terpotong di tengah
    # bisa lolos tanpa satu pun temuan. Panjangnya karena itu diperiksa di sini.
    if "% halaman hanya berisi diagram akor" not in text:
        n_bar = len(re.findall(r"\|", re.sub(r"^w:.*$", "", text, flags=re.M)))
        if n_bar < 8:
            bad.append(f"hanya {n_bar} garis birama, kemungkinan keluaran terpotong")

    return got, bad


def id_from_name(path: str) -> str | None:
    """Ambil id lagu dari nama berkas masukan, misalnya masuk/006_apa_saja.txt."""
    m = re.match(r"(\d+)", Path(path).stem)
    return str(int(m.group(1))) if m else None


def dataset_titles() -> dict[str, str]:
    """Judul menurut dataset, dipakai untuk memastikan berkas tidak tertukar."""
    if not CSV_PATH.exists():
        return {}
    csv.field_size_limit(10**8)
    with open(CSV_PATH, encoding="utf-8") as fh:
        return {(r.get("id") or "").strip(): (r.get("title") or "").strip()
                for r in csv.DictReader(fh)}


def scans_for(rid: str) -> list[str]:
    """Cari gambar milik satu id di direktori pindaian siap, urut halaman."""
    if not SCANS_DIR.is_dir():
        return []
    return sorted(
        (p.name for p in SCANS_DIR.glob(f"{rid}_*.jpg")),
        key=lambda n: (len(n), n),
    )


def stamp(text: str, rid: str, berkas: list[str]) -> str:
    """Sisipkan id dan daftar berkas, lalu samakan X: dengan id."""
    lines = text.splitlines()
    out = [lines[0]] if lines else ["%abc-2.1"]
    out.append(f"% laguqa-id {rid}")
    out.append(f"% laguqa-berkas {','.join(berkas) if berkas else '-'}")
    for line in lines[1:]:
        out.append(re.sub(r"^X:.*$", f"X:{rid}", line))
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="berkas teks keluaran model, atau - untuk stdin")
    ap.add_argument("--apply", action="store_true", help="tulis ke data/abc/")
    args = ap.parse_args()

    ABC_DIR.mkdir(parents=True, exist_ok=True)
    titles = dataset_titles()
    total_bad = 0
    n_stub = 0

    for name in args.files:
        raw = sys.stdin.read() if name == "-" else Path(name).read_text(encoding="utf-8")
        if not raw.strip() or STUB_MARK in raw.splitlines()[0]:
            n_stub += 1
            continue
        text = unfence(raw)
        got, bad = check_contract(text)

        rid = id_from_name(name)
        title = got.get("T", "?")
        print(f"\n=== {name}  ->  id {rid or '?'}  {title} ===")

        if rid is None:
            bad.append(
                f"nama berkas {Path(name).name!r} tidak diawali nomor lagu; "
                "beri nama seperti masuk/006.txt atau masuk/006_injit.txt"
            )
            berkas: list[str] = []
        else:
            berkas = scans_for(rid)
            if not berkas:
                bad.append(f"tidak ada gambar berawalan {rid}_ di {SCANS_DIR}")
            else:
                print(f"  gambar sumber: {', '.join(berkas)}")
            # Judul dibandingkan dengan dataset supaya keluaran lagu lain yang
            # tersimpan dengan nomor keliru langsung ketahuan, bukan baru
            # terlihat berbulan-bulan kemudian saat menyusun kunci jawaban.
            harap = titles.get(rid, "")
            if harap and title != "?":
                mirip = judul_mirip(title, harap)
                if mirip < MIRIP_MIN:
                    bad.append(
                        f"judul tidak cocok: berkas ini bernomor {rid} yang menurut "
                        f"dataset berjudul {harap!r}, tetapi isinya {title!r}"
                    )
                elif mirip < 1.0:
                    # Ejaan buku kerap berbeda dari ejaan dataset, dan bukan
                    # selalu bukunya yang keliru. Buku mencetak "BHINEKA
                    # TUNGGAL IKA" dengan satu N, sedangkan dataset memakai
                    # ejaan baku "Bhinneka". Selisih sekecil itu bukan tanda
                    # berkas tertukar, jadi dilaporkan tanpa menghalangi.
                    print(f"  [ejaan] buku {title!r}, dataset {harap!r}")

        for b in bad:
            print(f"  [KONTRAK] {b}")
        total_bad += len(bad)

        if rid is not None and not bad:
            text = stamp(text, rid, berkas)

        findings, stats = report(f"id {rid or '?'}", text, quiet=True)

        if bad or rid is None:
            print("  tidak disimpan, perbaiki pelanggaran kontrak dahulu")
            continue

        dst = ABC_DIR / f"{int(rid):03d}_{slugify(title)}.abc"
        if args.apply:
            dst.write_text(text, encoding="utf-8")
            print(f"  ditulis: {dst}")
        else:
            print(f"  akan ditulis: {dst}")

    if n_stub:
        print(f"\n{n_stub} berkas masih kosong, dilewati")
    print(f"\npelanggaran kontrak: {total_bad}")
    if not args.apply:
        print("Jalankan dengan --apply untuk menyimpan ke data/abc/")
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
