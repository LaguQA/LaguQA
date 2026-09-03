#!/usr/bin/env python3
"""Isi kolom dataset selain notasi dari kepala berkas ABC.

Enam kolom dataset sebenarnya sudah terbaca oleh model ketika mentranskripsi,
dan tersimpan di kepala berkas ABC:

    composer        <- C:
    origin          <- O:
    tempo           <- Q:
    time_signature  <- M:
    key_signature   <- % laguqa-do, cadangannya K:
    lyrics          <- gabungan baris w: dan W:

Mengetiknya ulang ke spreadsheet berarti membaca 107 gambar untuk kedua kalinya
dan membuka peluang salah salin. Nilainya ditarik dari berkas supaya kepala ABC
tetap satu-satunya tempat metadata ditulis.

Berbeda dari abc_sync.py yang menyalin notasi dari `data/abc/` setelah lulus
validator, skrip ini membaca direktori keluaran mentah apa adanya. Metadata
tidak bergantung pada benar tidaknya nada, sehingga tidak perlu menunggu berkas
lulus. Kolom yang sudah terisi tidak ditimpa kecuali diminta --timpa.

Mode bawaan dry-run. Tambahkan --apply untuk menulis.

Pemakaian:
    python scripts/08_fill_metadata.py
    python scripts/08_fill_metadata.py --apply
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from laguqa.notation.abc_ingest import STUB_MARK
from laguqa.paths import ABC_RAW_DIR, CSV_PATH

csv.field_size_limit(10**8)

KOLOM = [
    "composer", "origin", "key_signature", "time_signature",
    "time_signature_source", "tempo", "lyrics",
]

# Buku memakai dua gaya kepala halaman. Halaman bergaya "Do = C" mencantumkan
# biramanya ("4/4 Marcia"); halaman bergaya "1 = C" hanya mencantumkan tempo,
# tanpa birama sama sekali. Gaya kedua dipakai pada hampir seluruh lagu
# nasional, dan di situ birama pada M: adalah kesimpulan model dari hitungan
# bar, bukan angka yang terbaca.
#
# Asal-usulnya dicatat karena berpengaruh pada kunci jawaban: birama yang
# disimpulkan model tidak boleh dipakai sebagai jawaban benar untuk soal yang
# menguji model, sebab keduanya bersumber pada tebakan yang sama.
TEBAK_BIRAMA = "birama tidak tercetak"


def field(text: str, tag: str) -> str:
    m = re.search(rf"^{tag}:(.*)$", text, re.M)
    return m.group(1).strip() if m else ""


def directive(text: str, name: str) -> str:
    m = re.search(rf"^%%?[ \t]*{re.escape(name)}[ \t]+(.*)$", text, re.M)
    return m.group(1).strip() if m else ""


def kosong(v: str) -> str:
    """Field ABC yang tidak ada isinya ditulis '-' menurut kontrak prompt."""
    return "" if v.strip() in ("", "-") else v.strip()


def nada_dasar(text: str) -> tuple[str, str]:
    """Kembalikan (nada dasar, catatan).

    Buku menuliskan nada dasar sebagai "Do = C", dan model menyalinnya dengan
    ejaan yang berbeda-beda: "C", "1 = C", "1=C", "d". Semuanya diringkas
    menjadi satu huruf besar supaya kolomnya bisa dibandingkan mesin.

    K: bukan sumber utama meski isinya lebih rapi. K: adalah tanda mula yang
    ditulis model, sedangkan yang dicetak di buku adalah "Do =", dan itulah
    yang ditanyakan soal. Keduanya dibandingkan, selisihnya dilaporkan.
    """
    do, k = directive(text, "laguqa-do"), field(text, "K")
    kunci = re.sub(r"^\s*1\s*=\s*", "", do).strip()
    m = re.match(r"([A-Ga-g])\s*([#b]?)", kunci)
    if not m:
        if k:
            return k, f"do tidak terbaca ({do!r}), dipakai K:{k}"
        return "", f"do maupun K: tidak terbaca ({do!r})"
    hasil = m.group(1).upper() + m.group(2)
    if k and hasil != k:
        return hasil, f"do={hasil} tetapi K:{k}"
    return hasil, ""


def tempo(text: str) -> str:
    """Buang tanda kutip Q: tetapi pertahankan angka ketukan kalau ada."""
    q = kosong(field(text, "Q"))
    m = re.match(r'"([^"]*)"\s*(.*)$', q)
    if not m:
        return q
    kata, angka = m.group(1).strip(), m.group(2).strip()
    return f"{kata} ({angka})" if angka else kata


def potong(baris: str, awal: str = "") -> tuple[list[str], str]:
    """Susun ulang satu baris lirik ABC menjadi daftar kata.

    Baris w: ditulis per suku kata dan dipenuhi tanda yang mengatur
    penyelarasan dengan not, bukan bagian dari liriknya:

        -   suku kata masih satu kata dengan berikutnya
        _   not tambahan untuk suku kata sebelumnya, tidak ada teks baru
        *   not tanpa suku kata
        ~   dua kata dinyanyikan pada satu not
        \\-  tanda hubung yang memang tercetak
        |   penanda pindah bar

    "Ma-na ka-la be-ta sa-kit _ ha-ti" menjadi "Mana kala beta sakit hati".

    `awal` adalah potongan kata yang tergantung di akhir baris sebelumnya;
    kata pertama baris ini menjadi lanjutannya. Potongan yang masih tergantung
    di akhir baris ini dikembalikan sebagai nilai kedua, bukan dimasukkan ke
    daftar kata.
    """
    kata: list[str] = []
    kini = awal
    sambung = bool(awal)
    for tok in baris.split():
        tok = tok.replace("|", "")
        if not tok or set(tok) <= {"_", "*"}:
            continue
        tok = tok.replace("\\-", "\x00").replace("~", " ")
        lanjut = tok.endswith("-")
        tok = tok.rstrip("-").replace("-", "")
        if not tok:
            sambung = sambung or lanjut
            continue
        if sambung:
            kini += tok
        else:
            if kini:
                kata.append(kini)
            kini = tok
        sambung = lanjut
    if kini and not sambung:
        kata.append(kini)
        kini = ""
    return [k.replace("\x00", "-") for k in kata], kini


FIELD_RE = re.compile(r"^[A-Za-z]:")


def kelompok(text: str) -> tuple[list[list[str]], list[str]]:
    """Pisahkan baris lirik menjadi kelompok per baris musik, dan prosa W:.

    Baris w: menempel pada baris musik tepat di atasnya. Kalau satu baris
    musik diikuti tiga baris w:, ketiganya adalah bait pertama, kedua, dan
    ketiga untuk melodi yang sama, bukan tiga baris berurutan dari satu bait.
    """
    grup: list[list[str]] = []
    prosa: list[str] = []
    kini: list[str] | None = None
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("%"):
            continue
        if s.startswith("w:"):
            if kini is None:
                kini = []
                grup.append(kini)
            kini.append(s[2:])
        elif s.startswith("W:"):
            isi = s[2:].strip()
            if isi and isi != "-":
                prosa.append(isi)
        elif FIELD_RE.match(s):
            continue
        else:
            kini = None
    return grup, prosa


def lirik(text: str) -> str:
    """Susun ulang seluruh lirik, bait demi bait.

    Dua hal membuat penyusunannya tidak sesederhana menggabung baris w:.

    Pertama, bait ditumpuk secara menegak. Bait kedua sebuah lagu tersebar
    sebagai baris w: kedua di bawah setiap baris musik, bukan sebagai
    kumpulan baris di bagian bawah berkas. Membaca berkas dari atas ke bawah
    menganyam semua baitnya jadi satu.

    Kedua, pemenggalan baris musik tidak selalu jatuh di batas kata: "Garuda
    Pancasila A-" berlanjut ke "ku-lah" pada baris berikutnya. Potongan yang
    tergantung dibawa menyeberangi baris, dan kata utuhnya jatuh di baris
    tempat ia selesai.

    Baris W: huruf besar adalah bait tambahan yang dicetak sebagai prosa di
    bawah lagu, bukan per suku kata, jadi disalin apa adanya.
    """
    grup, prosa = kelompok(text)
    n = max((len(g) for g in grup), default=0)

    bait: list[str] = []
    for k in range(n):
        baris: list[str] = []
        sisa = ""
        for g in grup:
            if k < len(g):
                sumber = g[k]
            elif len(g) == 1:
                # Satu baris w: di bawah lagu yang bait lainnya bertumpuk
                # berarti bagian itu dinyanyikan sama pada setiap bait, jadi
                # ikut ke semuanya. Kelompok yang kurang lengkap tanpa alasan
                # itu dilewati daripada ditebak.
                sumber = g[0]
            else:
                continue
            kata, sisa = potong(sumber, sisa)
            if kata:
                baris.append(" ".join(kata))
        if sisa:
            # Bait berhenti di tengah kata. Tidak wajar, tetapi potongannya
            # tetap disimpan daripada hilang tanpa jejak.
            baris.append(sisa.replace("\x00", "-"))
        if baris:
            bait.append("\n".join(baris))

    return "\n\n".join(bait + prosa)


def curiga_gabung(teks: str) -> list[str]:
    """Cari kata hasil sambungan yang berkapital di tengah.

    Tanda hubung di akhir baris w: berarti katanya bersambung ke baris
    berikutnya. Model kerap membubuhkannya juga di tempat kata sudah selesai,
    sehingga dua kata menyatu: "ikhlas-" lalu "Da-ri" menjadi "ikhlasDari".

    Tidak diperbaiki otomatis. Kapital di tengah juga muncul pada sambungan
    yang benar ("di ta-ri-" lalu "Kan" menjadi "tariKan") dan pada kata yang
    memang ditulis begitu ("karuniaMu"), dan ketiganya tidak terbedakan tanpa
    kamus. Yang salah diperbaiki di berkas ABC-nya, bukan di sini.
    """
    return [w for w in teks.split() if re.search(r"[a-z][A-Z]", w)]


def baca(path: Path) -> tuple[dict[str, str], list[str]]:
    text = path.read_text(encoding="utf-8")
    kunci, catatan = nada_dasar(text)
    nilai = {
        "composer": kosong(field(text, "C")),
        "origin": kosong(field(text, "O")),
        "key_signature": kunci,
        "time_signature": kosong(field(text, "M")),
        "time_signature_source": "disimpulkan" if TEBAK_BIRAMA in text else "tercetak",
        "tempo": tempo(text),
        "lyrics": lirik(text),
    }
    pesan = [catatan] if catatan else []
    for k in ("time_signature", "lyrics"):
        if not nilai[k]:
            pesan.append(f"{k} kosong")
    gabung = curiga_gabung(nilai["lyrics"])
    if gabung:
        pesan.append("kata menyatu? " + " ".join(gabung))
    return nilai, pesan


def is_stub(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return not text.strip() or STUB_MARK in text.splitlines()[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None,
                    help=f"tempat berkas ABC dibaca (bawaan: {ABC_RAW_DIR})")
    ap.add_argument("--apply", action="store_true", help="tulis ke dataset")
    ap.add_argument("--timpa", action="store_true", help="timpa kolom yang sudah terisi")
    args = ap.parse_args()

    src = Path(args.dir) if args.dir else ABC_RAW_DIR
    if not src.is_dir():
        print(f"tidak ditemukan: {src}", file=sys.stderr)
        return 1
    with open(CSV_PATH, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys())
    if "time_signature_source" not in fields:
        fields.insert(fields.index("time_signature") + 1, "time_signature_source")
        for r in rows:
            r.setdefault("time_signature_source", "")

    berkas: dict[str, Path] = {}
    for p in sorted(src.glob("*.abc")):
        m = re.match(r"(\d+)_", p.name)
        if m and not is_stub(p):
            berkas[str(int(m.group(1)))] = p

    print(f"{'MODE APPLY' if args.apply else 'MODE DRY-RUN (tidak ada yang ditulis)'}")
    print(f"berkas terisi di {src}: {len(berkas)}\n")

    diisi = {k: 0 for k in KOLOM}
    dilewati = {k: 0 for k in KOLOM}
    hampa = {k: 0 for k in KOLOM}
    catatan: list[str] = []
    n_lagu = 0

    for r in rows:
        rid = (r.get("id") or "").strip()
        path = berkas.get(rid)
        if path is None:
            continue
        n_lagu += 1
        nilai, pesan = baca(path)
        for p in pesan:
            catatan.append(f"  {path.name}: {p}")
        for k in KOLOM:
            lama = (r.get(k) or "").strip()
            if lama and not args.timpa:
                if lama != nilai[k]:
                    dilewati[k] += 1
                continue
            if not nilai[k]:
                hampa[k] += 1
                continue
            r[k] = nilai[k]
            diisi[k] += 1

    if catatan:
        print("catatan:")
        print("\n".join(catatan))
        print()

    print(f"{'kolom':<16} {'diisi':>6} {'kosong di sumber':>18} {'sudah ada, dilewati':>21}")
    for k in KOLOM:
        print(f"{k:<16} {diisi[k]:>6} {hampa[k]:>18} {dilewati[k]:>21}")
    print(f"\n{n_lagu} lagu diproses")

    if args.apply:
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"ditulis: {CSV_PATH}")
    else:
        print("Jalankan dengan --apply untuk menulis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
