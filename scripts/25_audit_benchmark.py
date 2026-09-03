#!/usr/bin/env python3
"""Audit menyeluruh berkas benchmark sebelum dipakai atau diterbitkan.

Setiap pemeriksaan di sini lahir dari cacat yang benar-benar pernah lolos ke
angka hasil, atau dari kelas cacat yang sejenis dan belum pernah diperiksa.
Cacat benchmark tidak memberi tanda apa pun: skornya tetap terlihat wajar,
hanya salah. Itu sebabnya pemeriksaannya harus otomatis dan dijalankan ulang
setiap kali soal dibangun.

Keluaran memakai tiga tingkat:
  [GAGAL]     mengubah angka hasil, wajib diperbaiki sebelum dipakai
  [PERIKSA]   mencurigakan, perlu dilihat manusia
  [ok]        lulus

Pemakaian:
    python scripts/25_audit_benchmark.py
    python scripts/25_audit_benchmark.py --mc data/benchmark/laguqa_mc.jsonl
"""

from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from laguqa.paths import CSV_PATH  # noqa: E402

BENCH = Path(__file__).resolve().parents[1] / "data" / "benchmark"

gagal = 0
periksa = 0


def lapor(tingkat: str, judul: str, rincian: list[str] | None = None) -> None:
    global gagal, periksa
    if tingkat == "GAGAL":
        gagal += 1
    elif tingkat == "PERIKSA":
        periksa += 1
    tanda = {"GAGAL": "[GAGAL]  ", "PERIKSA": "[PERIKSA]", "ok": "[ok]     "}[tingkat]
    print(f"{tanda} {judul}")
    for r in (rincian or [])[:8]:
        print(f"            {r}")
    if rincian and len(rincian) > 8:
        print(f"            ... dan {len(rincian) - 8} lagi")


def muat(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def normal(s) -> str:
    return re.sub(r"[^\w]+", " ", str(s).lower()).strip()


# ---------------------------------------------------------------- pemeriksaan

def cek_opsi_kembar(items: list[dict]) -> None:
    """Dua opsi yang teksnya sama persis: soal punya dua jawaban benar."""
    kena = []
    for d in items:
        op = d.get("opsi") or {}
        nilai = [str(v).strip() for v in op.values()]
        if len(set(nilai)) != len(nilai):
            ganda = [v for v, n in collections.Counter(nilai).items() if n > 1]
            kena.append(f"{d['id']} [{d['kategori']}] kembar: {ganda}")
    lapor("GAGAL" if kena else "ok",
          f"opsi kembar persis ({len(kena)} soal)", kena)


def cek_opsi_bersarang(items: list[dict]) -> None:
    """Satu opsi termuat di dalam opsi lain, misalnya '5' di dalam \"5'\".

    Penilai memilih kecocokan terpanjang, jadi ini tidak selalu fatal, tetapi
    jawaban model yang hanya menyebut yang pendek jadi ambigu.
    """
    kena = []
    for d in items:
        op = d.get("opsi") or {}
        nilai = {k: str(v).strip() for k, v in op.items()}
        for (a, va), (b, vb) in itertools.permutations(nilai.items(), 2):
            if va and vb and va != vb and va.lower() in vb.lower():
                kena.append(f"{d['id']} [{d['kategori']}] {va!r} di dalam {vb!r}")
                break
    lapor("PERIKSA" if kena else "ok",
          f"opsi bersarang ({len(kena)} soal)", kena)


def kutipan(d: dict) -> str:
    """Bagian pertanyaan yang dikutip dari lagu, tanpa kalimat perintahnya.

    Kalimat perintah memuat kata umum seperti "yang" dan "lagu", sehingga
    mencocokkan seluruh pertanyaan menghasilkan banyak temuan palsu.
    """
    q = d.get("pertanyaan", "")
    return q.split("\n\n", 1)[1] if "\n\n" in q else q


def cek_kunci_di_pertanyaan(items: list[dict]) -> None:
    """Jawaban tercetak di dalam kutipan yang ditunjukkan pertanyaannya.

    Dipisah menurut kategori karena akibatnya berbeda, dan perbedaannya diukur
    bukan diduga. Pada `lirik_ke_judul` kebocorannya fatal: judul lagu
    Indonesia biasanya frasa yang diangkat dari liriknya sendiri, dan model
    tanpa pelatihan menjawab benar 26 dari 26 soal semacam itu sementara hanya
    48% pada sisanya. Pada `rumpang` kata jawabannya kadang terulang di baris
    yang sama, tetapi model tanpa pelatihan mendapat 40,0% pada soal itu dan
    40,4% pada sisanya, jadi tidak ada yang bisa dieksploitasi. Pengecohnya
    memang diambil dari kosakata lagu yang sama, sehingga terulangnya sebuah
    kata tidak menunjuk ke mana-mana.
    """
    berat, ringan = [], []
    for d in items:
        op = d.get("opsi") or {}
        jawab = str(op.get(d.get("kunci"), "")).strip()
        if len(jawab) > 3 and normal(jawab) and normal(jawab) in normal(kutipan(d)):
            baris = f"{d['id']} [{d['kategori']}] {jawab!r} ada di kutipan"
            (ringan if d["kategori"] == "rumpang" else berat).append(baris)
    lapor("GAGAL" if berat else "ok",
          f"kunci bocor di kutipan, kategori berbasis judul ({len(berat)} soal)",
          berat)
    lapor("PERIKSA" if ringan else "ok",
          f"kata kunci terulang di kutipan rumpang ({len(ringan)} soal, "
          f"terukur tidak memberi keuntungan)", ringan[:3])


def cek_petunjuk_bentuk(items: list[dict]) -> None:
    """Bisakah kunci ditebak dari bentuk opsinya, tanpa membaca isinya?

    Dua jalan pintas yang paling mungkin: memilih opsi terpendek atau
    terpanjang, dan memilih yang berhuruf kapital. Kalau salah satunya jauh di
    atas peluang acak, yang terukur tata letak soal, bukan pengetahuan.
    """
    n = len(items)
    if not n:
        return
    acak = 100 / 5
    pendek = sum(1 for d in items
                 if min(d["opsi"], key=lambda k: (len(str(d["opsi"][k])), k))
                 == d["kunci"]) / n * 100
    panjang = sum(1 for d in items
                  if max(d["opsi"], key=lambda k: (len(str(d["opsi"][k])), k))
                  == d["kunci"]) / n * 100
    kena = [f"pilih terpendek: {pendek:.1f}% (acak {acak:.0f}%)",
            f"pilih terpanjang: {panjang:.1f}% (acak {acak:.0f}%)"]
    buruk = max(pendek, panjang) > acak + 5
    lapor("GAGAL" if buruk else "ok",
          "kunci dapat ditebak dari panjang opsi", kena)

    besar_kunci = sum(1 for d in items
                      if str(d["opsi"][d["kunci"]]).strip()[:1].isupper()) / n * 100
    lain = [str(v).strip() for d in items for h, v in d["opsi"].items()
            if h != d["kunci"]]
    besar_lain = sum(1 for v in lain if v[:1].isupper()) / len(lain) * 100
    selisih = abs(besar_kunci - besar_lain)
    lapor("GAGAL" if selisih > 10 else "ok",
          f"huruf kapital: kunci {besar_kunci:.1f}% vs pengecoh "
          f"{besar_lain:.1f}% (selisih {selisih:.1f})")


def cek_kunci_sah(items: list[dict]) -> None:
    """Huruf kunci harus ada di antara opsinya."""
    kena = [f"{d['id']} kunci={d.get('kunci')!r} opsi={sorted((d.get('opsi') or {}))}"
            for d in items if d.get("kunci") not in (d.get("opsi") or {})]
    lapor("GAGAL" if kena else "ok", f"kunci di luar opsi ({len(kena)} soal)", kena)


def cek_sebaran_huruf(items: list[dict]) -> None:
    """Kunci tidak boleh menumpuk di satu huruf: bisa ditebak tanpa membaca."""
    c = collections.Counter(d.get("kunci") for d in items if d.get("kunci"))
    n = sum(c.values())
    if not n:
        return
    harap = 1 / len(c)
    ekstrem = [f"{h}: {v} ({v / n * 100:.1f}%)" for h, v in sorted(c.items())
               if abs(v / n - harap) > 0.05]
    lapor("PERIKSA" if ekstrem else "ok",
          f"sebaran huruf kunci, harapan {harap * 100:.0f}% per huruf",
          ekstrem or [f"{h}: {v / n * 100:.1f}%" for h, v in sorted(c.items())])


def cek_ruang_jawaban(items: list[dict]) -> None:
    """Kategori yang jawabannya hanya sedikit nilai berbeda.

    `hitung_bar` pernah bisa dijawab 100% tanpa membaca notasi karena
    jawabannya cuma {2, 4, 8}.
    """
    per = collections.defaultdict(list)
    for d in items:
        op = d.get("opsi") or {}
        per[d["kategori"]].append(str(op.get(d.get("kunci"), "")).strip())
    kena = []
    for kat, nilai in sorted(per.items()):
        beda = len(set(nilai))
        tersering = collections.Counter(nilai).most_common(1)[0]
        bagian = tersering[1] / len(nilai) * 100
        if beda <= 3 or bagian > 75:
            kena.append(f"{kat}: {beda} nilai berbeda dari {len(nilai)} soal, "
                        f"tersering {tersering[0]!r} {bagian:.0f}%")
    lapor("PERIKSA" if kena else "ok",
          f"ruang jawaban sempit ({len(kena)} kategori)", kena)


def cek_soal_kembar(items: list[dict]) -> None:
    """Pertanyaan yang sama persis muncul lebih dari sekali."""
    c = collections.Counter(normal(d.get("pertanyaan", "")) for d in items)
    kena = [f"{n}x: {q[:70]}" for q, n in c.most_common() if n > 1]
    lapor("PERIKSA" if kena else "ok",
          f"pertanyaan berulang ({len(kena)} bentuk)", kena)


def cek_entitas_kembar(items: list[dict], alias: list[set[str]]) -> None:
    """Dua opsi yang merujuk hal yang sama meski tulisannya beda."""
    def kel(v):
        v = str(v).strip()
        for g in alias:
            if v in g:
                return frozenset(g)
        return v
    kena = []
    for d in items:
        op = d.get("opsi") or {}
        g = [kel(v) for v in op.values()]
        if len(set(g)) < len(g):
            kena.append(f"{d['id']} [{d['kategori']}] {list(op.values())}")
    lapor("GAGAL" if kena else "ok",
          f"opsi merujuk entitas yang sama ({len(kena)} soal)", kena)


def cek_birama_disimpulkan(items: list[dict], sumber: dict[str, str]) -> None:
    """Soal birama tidak boleh dibuat dari 50 lagu yang biramanya disimpulkan."""
    kena = [f"{d['id']} lagu {d['id_lagu']}" for d in items
            if d.get("kategori") == "birama"
            and sumber.get(d.get("id_lagu")) != "tercetak"]
    lapor("GAGAL" if kena else "ok",
          f"soal birama dari birama yang disimpulkan ({len(kena)} soal)", kena)


def cek_judul_ambigu(items: list[dict], judul: dict[str, str]) -> None:
    """Dua Desaku berjudul sama: soal berbasis judul jadi ambigu."""
    hitung = collections.Counter(judul.values())
    ganda = {j for j, n in hitung.items() if n > 1}
    kena = []
    for d in items:
        op = d.get("opsi") or {}
        for v in op.values():
            if str(v).strip() in ganda:
                kena.append(f"{d['id']} [{d['kategori']}] opsi {v!r} ambigu")
                break
    lapor("PERIKSA" if kena else "ok",
          f"opsi memakai judul yang tidak unik ({len(kena)} soal)", kena)


def cek_kunci_cocok_csv(items: list[dict], baris: dict[str, dict]) -> None:
    """Kunci soal fakta harus sama dengan isi CSV, bukan karangan."""
    kolom = {"pencipta": "composer", "asal": "origin", "tempo": "tempo",
             "nada_dasar": "key_signature", "birama": "time_signature"}
    kena = []
    for d in items:
        k = kolom.get(d.get("kategori"))
        if not k:
            continue
        r = baris.get(d.get("id_lagu"))
        if not r:
            continue
        jawab = str((d.get("opsi") or {}).get(d.get("kunci"), "")).strip()
        benar = (r.get(k) or "").strip()
        if k == "key_signature":
            benar = f"Do = {benar}"
        if benar and normal(jawab) != normal(benar):
            kena.append(f"{d['id']} [{d['kategori']}] lagu {d['id_lagu']}: "
                        f"kunci {jawab!r} != csv {benar!r}")
    lapor("GAGAL" if kena else "ok",
          f"kunci tidak cocok dengan CSV ({len(kena)} soal)", kena)


def cek_bocor_latih_uji(train: Path, test: Path) -> None:
    """Tidak ada lagu yang muncul di kedua sisi berkas split."""
    if not train.exists() or not test.exists():
        lapor("ok", "berkas split tidak ada, pemeriksaan kebocoran dilewati")
        return
    a = {d.get("id_lagu") for d in muat(train)}
    b = {d.get("id_lagu") for d in muat(test)}
    iris = sorted(a & b, key=lambda x: int(x) if str(x).isdigit() else 0)
    lapor("GAGAL" if iris else "ok",
          f"lagu muncul di sisi latih dan uji ({len(iris)} lagu)",
          [f"lagu {x}" for x in iris])


def cek_jumlah_opsi(items: list[dict], n: int) -> None:
    kena = [f"{d['id']} punya {len(d.get('opsi') or {})} opsi" for d in items
            if len(d.get("opsi") or {}) != n]
    lapor("GAGAL" if kena else "ok",
          f"soal tanpa tepat {n} opsi ({len(kena)} soal)", kena)


def cek_opsi_kosong(items: list[dict]) -> None:
    kena = [f"{d['id']} opsi {h} kosong" for d in items
            for h, v in (d.get("opsi") or {}).items() if not str(v).strip()]
    lapor("GAGAL" if kena else "ok", f"opsi kosong ({len(kena)})", kena)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mc", type=Path, default=BENCH / "laguqa_mc.jsonl")
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    args = ap.parse_args()

    items = [d for d in muat(args.mc) if "kategori" in d]
    with open(args.csv, encoding="utf-8") as fh:
        baris = {r["id"]: r for r in csv.DictReader(fh)}

    print(f"berkas   : {args.mc}")
    print(f"soal     : {len(items)}")
    print(f"kategori : {len(set(d['kategori'] for d in items))}")
    print(f"csv      : {args.csv} ({len(baris)} lagu)\n")

    alias = [
        {"Ismail Marzuki", "Imail Marzuki", "Ismail, MZ"},
        {"NN", "NN."},
        {"Mochtar Embut", "Mukhtar Embut"},
        {"Alfred Simanjuntak", "Alfred Simanjunatak"},
    ]
    cek_jumlah_opsi(items, 5)
    cek_opsi_kosong(items)
    cek_kunci_sah(items)
    cek_opsi_kembar(items)
    cek_entitas_kembar(items, alias)
    cek_kunci_di_pertanyaan(items)
    cek_kunci_cocok_csv(items, baris)
    cek_birama_disimpulkan(
        items, {i: (r.get("time_signature_source") or "").strip()
                for i, r in baris.items()})
    cek_judul_ambigu(items, {i: (r.get("title") or "").strip()
                             for i, r in baris.items()})
    cek_petunjuk_bentuk(items)
    cek_opsi_bersarang(items)
    cek_ruang_jawaban(items)
    cek_soal_kembar(items)
    cek_sebaran_huruf(items)
    cek_bocor_latih_uji(BENCH / "laguqa_train_split70.jsonl",
                        BENCH / "laguqa_test_split37.jsonl")

    print(f"\n{gagal} GAGAL, {periksa} PERIKSA")
    return 1 if gagal else 0


if __name__ == "__main__":
    sys.exit(main())
