#!/usr/bin/env python3
"""Assemble the publishable half of LaguQA into a folder ready for the Hub.

The benchmark, not the adapters, is what this thesis contributes: an adapter is
one model's weights, while the benchmark is what lets anyone measure any model
on the same 107 songs. So it gets the same treatment -- a card written from the
data rather than about it, and a manifest that makes silent drift detectable.

WHAT IS DELIBERATELY LEFT OUT

The scanned pages. The book is from 2025 and still fully in copyright, so the
1.06 GB under sumber/ never leaves this machine. What ships is metadata,
transcriptions made for analysis, and short excerpts inside questions. Anyone
who wants to check a transcription against the printed page can: every song
carries its page number, and abc_to_jianpu turns the ABC back into the notasi
angka the book prints.

Copying rather than symlinking, so the released folder is a snapshot that
cannot change underneath a published hash.

Usage:
    python scripts/18_dataset_release.py
    python scripts/18_dataset_release.py --out rilis-dataset --apply
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from laguqa.paths import ABC_DIR, BENCHMARK_DIR, CSV_PATH, DATA_DIR, REPO_ROOT

csv.field_size_limit(10**8)

#: Kartu bahasa Inggris. Berkas inilah yang ditampilkan Hugging Face di halaman
#: dataset, jadi hanya berkas ini yang memuat metadata YAML. Medan `language: id`
#: menerangkan bahasa datanya, bukan bahasa kartunya.
CARD_EN = """---
license: cc-by-nc-4.0
language:
  - id
tags:
  - music
  - indonesian
  - question-answering
  - benchmark
pretty_name: LaguQA
size_categories:
  - 1K<n<10K
---

# LaguQA

A benchmark for what a language model knows about {n_songs} Indonesian regional
and national songs. Every one was transcribed by hand from a printed songbook
into ABC 2.1 notation, and every value traces back to a specific page.

The questions cover bibliographic facts such as composer and region of origin,
and reasoning over the notation. Some show a fragment of number notation and ask
which song it is. Others ask the model to count bars or name the highest note.
Number notation is how music is taught across Indonesia, and no benchmark the
author could find tests whether a model can read it.

Part of an undergraduate thesis in Informatics, Universitas Ahmad Dahlan.

[Baca dalam bahasa Indonesia](README.id.md)

| | |
|---|---|
| Code | [github.com/IRedDragonICY/LaguQA](https://github.com/IRedDragonICY/LaguQA) |
| Demo | [IRedDragonICY/LaguQA-Demo](https://huggingface.co/spaces/IRedDragonICY/LaguQA-Demo) |
| Model | [IRedDragonICY/LaguQA-Gemma4-E2B](https://huggingface.co/IRedDragonICY/LaguQA-Gemma4-E2B) |
| Mirror | [kaggle.com/datasets/ireddragonicy/laguqa](https://www.kaggle.com/datasets/ireddragonicy/laguqa) |

The demo draws and plays each song's notation from the ABC files in this
repository, so what a visitor hears is the same material the notation questions
are keyed against.

## Contents

| file | rows | contents |
|---|---|---|
{files_table_en}

There are {n_categories} question categories, from metadata facts to reasoning
over notation.

## Score floor

Accuracy on this benchmark means nothing measured against zero. The book is
dominated by 4/4 time and the key of Do = C, so a guesser that always answers
the most common value already collects most of the metadata questions without
knowing a single song.

The two test files have separate floors, and the numbers are not
interchangeable, because the questions differ.

| control | `laguqa_test_split37` | `laguqa_test` |
|---|---|---|
| always the most common answer | **26.9%** | **33.6%** |
| random within the same category | 24.3% | 28.6% |
| answering nothing at all | 0.0% | 0.0% |

The numbers are lenient. All three can be regenerated with
`scripts/17_controls.py` and are scored by the same scorer used on real models.
For comparison, Gemma-4-E2B without training scores 19.6% on `split37`, below a
guesser that knows no song at all.

## Contamination check

The multiple-choice file carries a canary string on its first line. A model that
can reproduce that string was trained on this file, which invalidates its score.
A canary stays useful after the data is public.

## Limitations

- **{n_meter_inferred} songs have an inferred time signature rather than a
  printed one.** Pages in the `1 = C` style print no time signature. The
  `time_signature_source` column marks them, and no time-signature question is
  generated from those songs.
- **{n_raw} ABC files are unverified**, so notation-reasoning questions reach
  only {n_verified} songs.
- **Composer is filled for {n_composer} of {n_songs} rows, region of origin for
  {n_origin} of {n_songs}.** The book names a composer for national songs and a
  region for regional ones, almost never both. Those gaps are not missing data.
  The correct answer for such a row is that the book does not state it, and
  abstention questions exist to test exactly that.
- **Eighteen words in the lyrics are still joined wrongly**, because one
  syllable was transcribed capitalised, as in `IndoNesia`. The lyric keys for
  those words carry the wrong spelling, so a model that gets them right is
  marked wrong. The list is in `docs/lirik-perlu-dicek.md`.
- **Two songs share a title.** The book contains two different songs called
  "Desaku", with different melodies and composers. The `title_unique` column
  separates them, and is used only when a question has to say which one it
  means.

`SOURCE.md`, included here, has the details.

## Licence

The metadata and transcriptions here are released under CC BY-NC 4.0. The source
book remains fully copyrighted and its scans are not included. Indonesian Law
28/2014 Article 44 permits quotation for research and education as long as the
source is named, and what is published here is designed to stay within that.

Two different layers of rights apply. The book is a compilation: what its
publisher holds is the selection and arrangement of the contents, along with the
page layout. Each song inside has its own rights status, separate from the book.
That is why page scans are never published, while lyrics and melodies are
treated according to each song's status.

The `rights_class` column records the class of each song, and `HAK-CIPTA.md`
explains the basis along with how to file an objection. That file lays out the
reasoning behind the classification. It is not legal advice, and it is not a
claim that everything in this dataset is free to use.

## Citation

Citing this dataset requires citing the source book as well.

```bibtex
@book{{pustakabaru2025,
  author    = {{{{Tim Pustaka Baru}}}},
  title     = {{Koleksi Lengkap Lagu-Lagu Daerah \\& Wajib Nasional}},
  publisher = {{Pustaka Baru Press}},
  address   = {{Banguntapan, Bantul, Yogyakarta}},
  year      = {{2025}},
  isbn      = {{978-602-0874-22-7}},
  pages     = {{192}}
}}
```
"""

#: Kartu bahasa Indonesia. Tanpa metadata YAML karena Hugging Face hanya membaca
#: metadata dari README.md.
CARD_ID = """# LaguQA

Benchmark untuk menguji apa yang model bahasa tahu tentang {n_songs} lagu daerah
dan wajib nasional Indonesia. Seluruhnya ditranskripsi tangan dari satu buku
cetak ke notasi ABC 2.1, dan tiap nilai bisa ditelusuri ke satu halaman buku.

Soalnya mencakup fakta bibliografis seperti pencipta dan daerah asal, dan juga
penalaran atas notasinya. Sebagian soal memberi potongan notasi angka lalu
meminta model menyebut lagunya, sebagian lagi meminta model menghitung bar atau
menyebut nada tertinggi. Notasi angka dipakai luas dalam pengajaran musik di
Indonesia, sedangkan tolok ukur yang menguji pembacaannya belum penulis temukan.

Bagian dari penelitian skripsi Program Studi Informatika, Universitas Ahmad
Dahlan.

[Read this in English](README.md)

| | |
|---|---|
| Kode | [github.com/IRedDragonICY/LaguQA](https://github.com/IRedDragonICY/LaguQA) |
| Demo | [IRedDragonICY/LaguQA-Demo](https://huggingface.co/spaces/IRedDragonICY/LaguQA-Demo) |
| Model | [IRedDragonICY/LaguQA-Gemma4-E2B](https://huggingface.co/IRedDragonICY/LaguQA-Gemma4-E2B) |
| Cermin | [kaggle.com/datasets/ireddragonicy/laguqa](https://www.kaggle.com/datasets/ireddragonicy/laguqa) |

Demonya menggambar dan membunyikan notasi tiap lagu dari berkas ABC di
repositori ini, sehingga yang terdengar di sana sama dengan yang menjadi kunci
jawaban soal notasi.

## Isi

| berkas | baris | isi |
|---|---|---|
{files_table}

Ada {n_categories} kategori soal, dari fakta metadata sampai penalaran notasi.

## Batas bawah skor

Akurasi di benchmark ini tidak berarti apa-apa bila dibandingkan dengan nol.
Buku ini didominasi birama 4/4 dan nada dasar Do = C, jadi penebak yang selalu
menjawab nilai tersering sudah mendapat sebagian besar soal metadata tanpa
mengenal satu lagu pun.

Kedua berkas uji punya batas bawahnya sendiri, dan angkanya tidak bisa
dipertukarkan karena soalnya berbeda.

| kontrol | `laguqa_test_split37` | `laguqa_test` |
|---|---|---|
| selalu jawaban tersering | **26,9%** | **33,6%** |
| acak dari kategori sama | 24,3% | 28,6% |
| tidak menjawab apa pun | 0,0% | 0,0% |

Angka toleran. Ketiganya dapat dibangkitkan ulang dengan
`scripts/17_controls.py` dan dinilai memakai penilai yang sama seperti model
sungguhan. Sebagai pembanding, Gemma-4-E2B tanpa pelatihan mencetak 19,6% pada
`split37`, di bawah penebak yang tidak mengenal satu lagu pun.

## Deteksi kontaminasi

Berkas pilihan ganda memuat untai penanda di baris pertama. Model yang pernah
menghasilkan untai itu berarti dilatih memakai berkas ini, sehingga skornya
tidak sah. Penanda semacam ini tetap berguna setelah datanya terbuka.

## Batasan

- **{n_meter_inferred} lagu biramanya disimpulkan, bukan dibaca.** Halaman
  bergaya `1 = C` tidak mencetak tanda birama. Kolom `time_signature_source`
  menandainya, dan tidak ada soal birama yang dibangkitkan dari lagu-lagu itu.
- **{n_raw} berkas ABC belum terverifikasi**, sehingga soal penalaran notasi
  hanya menjangkau {n_verified} lagu.
- **Pencipta terisi {n_composer} dari {n_songs} baris, asal daerah
  {n_origin} dari {n_songs}.** Buku mencantumkan pencipta untuk lagu nasional
  dan asal daerah untuk lagu daerah, hampir tidak pernah keduanya. Kekosongan
  itu bukan data yang hilang. Jawaban yang benar untuk baris tersebut adalah
  bahwa buku tidak mencantumkannya, dan soal abstain dibuat untuk mengujinya.
- **Delapan belas kata di lirik masih tergabung salah** akibat satu suku kata
  tertranskripsi berhuruf kapital, misalnya `IndoNesia`. Kunci lirik untuk
  kata-kata itu memuat ejaan keliru, jadi model yang benar justru dinilai
  meleset. Daftarnya ada di `docs/lirik-perlu-dicek.md`.
- **Dua lagu berjudul sama.** Buku memuat dua "Desaku" yang berbeda melodi dan
  pencipta. Kolom `title_unique` membedakannya, dan hanya dipakai saat
  pertanyaan harus menyebut lagu mana yang dimaksud.

Rinciannya di `SOURCE.md`, yang ikut disertakan.

## Lisensi

Metadata dan transkripsi di sini dirilis CC BY-NC 4.0. Buku sumbernya tetap
berhak cipta penuh dan pindaiannya tidak disertakan. UU 28/2014 Pasal 44
membolehkan pengutipan untuk penelitian dan pendidikan sepanjang sumbernya
disebut; yang diterbitkan di sini dirancang agar tetap di dalam batas itu.

Ada dua lapis hak yang berbeda di sini. Buku ini kompilasi: yang dimiliki
penerbitnya adalah pemilihan dan penyusunan isinya, beserta tata letak
halamannya. Tiap lagu di dalamnya punya status hak sendiri, terpisah dari
bukunya. Karena itu pindaian halaman tidak pernah diterbitkan, sedangkan lirik
dan melodi diperlakukan menurut status tiap lagu.

Kolom `rights_class` mencatat golongan tiap lagu, dan `HAK-CIPTA.md` menjelaskan
dasarnya beserta cara mengajukan keberatan. Berkas itu memaparkan dasar
penggolongannya. Isinya bukan nasihat hukum, dan bukan pernyataan bahwa seluruh
isi dataset ini bebas dipakai.

## Sitasi

Sitasi dataset ini perlu disertai sitasi buku sumbernya.

```bibtex
@book{{pustakabaru2025,
  author    = {{{{Tim Pustaka Baru}}}},
  title     = {{Koleksi Lengkap Lagu-Lagu Daerah \\& Wajib Nasional}},
  publisher = {{Pustaka Baru Press}},
  address   = {{Banguntapan, Bantul, Yogyakarta}},
  year      = {{2025}},
  isbn      = {{978-602-0874-22-7}},
  pages     = {{192}}
}}
```

```bibtex
@misc{{laguqa,
  author = {{Hendianto, Mohammad Farid}},
  title  = {{LaguQA: benchmark pengetahuan lagu Indonesia untuk model bahasa}},
  year   = {{2026}},
  note   = {{Skripsi, Program Studi Informatika, Universitas Ahmad Dahlan}}
}}
```
"""

DESCRIBE = {
    "laguqa.csv": "tabel utama, satu baris satu lagu",
    "laguqa_train.jsonl": "latih, seluruh 107 lagu (untuk model rilis)",
    "laguqa_test.jsonl": "uji pasangan regime penuh",
    "laguqa_train_split70.jsonl": "latih, 70 lagu (untuk percobaan)",
    "laguqa_test_split37.jsonl": "uji, 37 lagu yang tidak pernah dilatih",
    "laguqa_mc.jsonl": ("pilihan ganda A-E, dasar tebakan acak 20%; pada "
                        "kategori birama 25% karena satu opsinya tanda birama "
                        "yang tidak dipakai buku ini"),
    "split.json": "pembagian 70/37 yang dibekukan",
    "daftar-isi.csv": "daftar isi buku, untuk memeriksa nomor halaman",
    "laguqa_manifest.json": "SHA-256 keempat berkas terbuka",
    "laguqa_mc_manifest.json": "SHA-256 berkas pilihan ganda",
}

DESCRIBE_EN = {
    "laguqa.csv": "the main table, one row per song",
    "laguqa_train.jsonl": "training, all 107 songs (used for the released model)",
    "laguqa_test.jsonl": "test, full-regime pairs",
    "laguqa_train_split70.jsonl": "training, 70 songs (used for the experiments)",
    "laguqa_test_split37.jsonl": "test, 37 songs never trained on",
    "laguqa_mc.jsonl": ("multiple choice A-E, random guessing floor 20%; 25% in "
                        "the time-signature category, where one option is a "
                        "signature the book never uses"),
    "split.json": "the frozen 70/37 split",
    "daftar-isi.csv": "the book's table of contents, for checking page numbers",
    "laguqa_manifest.json": "SHA-256 of the four open files",
    "laguqa_mc_manifest.json": "SHA-256 of the multiple-choice file",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def records_in(path: Path) -> int:
    """Rows of data, not physical lines.

    The lyrics column holds embedded newlines inside quoted fields, so counting
    lines reports 4063 songs for a table of 107. And laguqa_mc.jsonl opens with
    a canary/version header that is not a question. Both would have gone into
    the published card as fact.
    """
    if path.suffix == ".csv":
        with open(path, encoding="utf-8", newline="") as fh:
            return sum(1 for _ in csv.reader(fh)) - 1  # minus the header row
    if path.suffix == ".jsonl":
        n = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                # Only laguqa_mc.jsonl has a header, and it is the one line
                # carrying the canary. Everything else is an item. Keying off
                # "id" instead was wrong: the open-ended items do not have one,
                # so all four of those files reported zero questions.
                n += "canary" not in json.loads(line)
        return n
    return 1


def tulis_hak(out: Path) -> None:
    """Tambahkan kolom rights_class ke CSV rilis dan tulis HAK-CIPTA.md.

    Digolongkan dari data yang memang ada, bukan dari penilaian hukum. Yang
    menentukan cuma satu hal yang terbaca di CSV: apakah bukunya menyebut nama
    pencipta. Tahun wafat pencipta tidak ada di dataset, jadi masa berlaku
    haknya tidak diklaim di sini -- pembaca diberi namanya dan dasarnya,
    lalu memeriksa sendiri.
    """
    csv_out = out / "laguqa.csv"
    with open(csv_out, encoding="utf-8", newline="") as fh:
        r = csv.DictReader(fh)
        fields = list(r.fieldnames or [])
        rows = list(r)

    if "rights_class" not in fields:
        fields.append("rights_class")
    for x in rows:
        x["rights_class"] = ("pencipta dicantumkan"
                             if (x.get("composer") or "").strip()
                             else "pencipta tidak dicantumkan")
    with open(csv_out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    bernama = sum(1 for x in rows if (x.get("composer") or "").strip())
    tanpa = len(rows) - bernama
    daerah = sum(1 for x in rows if not (x.get("composer") or "").strip()
                 and (x.get("song_type") or "").strip() == "Daerah")
    pencipta = sorted({(x.get("composer") or "").strip() for x in rows
                       if (x.get("composer") or "").strip()})

    baris = "\n".join(f"| {p} | "
                      + ", ".join(x["title"] for x in rows
                                  if (x.get("composer") or "").strip() == p)
                      + " |" for p in pencipta)

    (out / "HAK-CIPTA.md").write_text(f"""# Status hak per lagu

Berkas ini keterbukaan, bukan nasihat hukum dan bukan pernyataan bahwa seluruh
isi rilis ini bebas dipakai untuk keperluan apa pun. Isinya menjelaskan apa
yang diterbitkan, apa yang ditahan, dan atas dasar apa, supaya pemegang hak
yang berkeberatan tahu persis apa yang harus dipersoalkan.

## Dua lapis yang tidak boleh tercampur

Sumbernya sebuah **kompilasi** terbitan 2025. Yang dimiliki penerbit adalah
pemilihan dan penyusunan isinya beserta tata letak halamannya. Itu sebabnya
**tidak satu pun pindaian halaman diterbitkan**, dan tidak akan.

Lagu-lagu di dalamnya punya status sendiri, terpisah dari bukunya. Notasi ABC
di rilis ini adalah transkripsi yang dibuat untuk penelitian ini, bukan salinan
tata letak buku.

## Golongan

| golongan | jumlah | keterangan |
|---|---|---|
| pencipta tidak dicantumkan | {tanpa} | {daerah} di antaranya lagu daerah. Lagu rakyat yang penciptanya tidak diketahui termasuk ekspresi budaya tradisional, dipegang negara menurut UU 28/2014 Pasal 38, dengan kewajiban menyebut asal dan menjaga kepatutan. |
| pencipta dicantumkan | {bernama} | Hak ciptanya melekat pada penciptanya. UU 28/2014 Pasal 58 memberi perlindungan seumur hidup pencipta ditambah 70 tahun, sehingga statusnya berbeda-beda menurut tahun wafat masing-masing. Dataset ini tidak memuat tahun wafat dan tidak mengklaim status apa pun per lagu. |

Kolom `rights_class` di `laguqa.csv` mencatat golongan tiap baris. Kolom
`composer_printed` menyimpan ejaan persis seperti tercetak di buku, sedangkan
`composer` sudah dibakukan.

## Dasar penerbitan

UU 28/2014 Pasal 44 membolehkan penggunaan untuk pendidikan dan penelitian
sepanjang sumbernya disebut dan tidak merugikan kepentingan wajar pencipta.
Rilis ini nirlaba, dilisensikan CC BY-NC 4.0, dan disusun untuk keperluan
penelitian bahasa.

## Pencipta yang tercantum

| pencipta | lagu |
|---|---|
{baris}

## Keberatan

Pemegang hak yang berkeberatan atas lagu tertentu dapat mengajukan penarikan
lewat halaman diskusi repositori ini. Lagu yang dipersoalkan akan dikeluarkan
dari rilis berikutnya beserta seluruh soal yang dibangun darinya, dan
manifesnya diperbarui sehingga penarikan itu terlacak.
""", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("rilis-dataset"))
    ap.add_argument("--apply", action="store_true", help="tulis berkasnya")
    args = ap.parse_args(argv)

    with open(CSV_PATH, encoding="utf-8") as fh:
        songs = list(csv.DictReader(fh))

    wanted = [CSV_PATH, DATA_DIR / "split.json", DATA_DIR / "daftar-isi.csv"]
    wanted += sorted(BENCHMARK_DIR.glob("*.jsonl"))
    wanted += sorted(BENCHMARK_DIR.glob("*_manifest.json"))
    present = [p for p in wanted if p.exists()]
    for p in wanted:
        if not p.exists():
            print(f"  [lewat] tidak ada: {p}")

    abc_files = sorted(ABC_DIR.glob("*.abc"))
    status = Counter(s["abc_status"] for s in songs)
    n_verified = status.get("terverifikasi", 0)

    rows, rows_en = [], []
    for p in present:
        n = records_in(p)
        jumlah = n if n > 1 else ""
        rows.append(f"| `{p.name}` | {jumlah} | {DESCRIBE.get(p.name, '')} |")
        rows_en.append(f"| `{p.name}` | {jumlah} | {DESCRIBE_EN.get(p.name, '')} |")
    rows.append(f"| `abc/` | {len(abc_files)} | transkripsi ABC 2.1, satu berkas satu lagu |")
    rows.append("| `SOURCE.md` | | asal data, cara verifikasi, dan batasannya |")
    rows_en.append(f"| `abc/` | {len(abc_files)} | ABC 2.1 transcriptions, one file per song |")
    rows_en.append("| `SOURCE.md` | | where the data comes from, how it was verified, and its limits |")

    angka = dict(
        n_songs=len(songs),
        n_categories=len({x["kategori"] for x in (
            json.loads(l) for l in
            (BENCHMARK_DIR / "laguqa_test_split37.jsonl").read_text(
                encoding="utf-8").splitlines() if l.strip())}),
        n_meter_inferred=sum(1 for s in songs
                             if s["time_signature_source"] != "tercetak"),
        n_raw=len(songs) - n_verified,
        n_verified=n_verified,
        n_composer=sum(1 for s in songs if s["composer"].strip() not in {"", "-"}),
        n_origin=sum(1 for s in songs if s["origin"].strip() not in {"", "-"}),
    )
    card = CARD_EN.format(files_table_en="\n".join(rows_en), **angka)
    card_id = CARD_ID.format(files_table="\n".join(rows), **angka)

    print(f"\n{len(present)} berkas data + {len(abc_files)} berkas ABC")
    if not args.apply:
        print(f"\npratinjau. tambahkan --apply untuk menulis ke {args.out}/")
        print("\n--- README.md ---")
        print(card[:1400])
        print("\n--- README.id.md ---")
        print(card_id[:600])
        return 0

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "abc").mkdir(exist_ok=True)

    manifest: dict[str, str] = {}
    for p in present:
        shutil.copy2(p, out / p.name)
        manifest[p.name] = sha256(p)
        print(f"  {p.name:34} {sha256(p)[:16]}")
    for p in abc_files:
        shutil.copy2(p, out / "abc" / p.name)
    manifest["abc/"] = hashlib.sha256(
        "".join(sha256(p) for p in abc_files).encode()).hexdigest()

    source = REPO_ROOT.parent / "SOURCE.md"
    if source.exists():
        shutil.copy2(source, out / "SOURCE.md")
        manifest["SOURCE.md"] = sha256(source)

    tulis_hak(out)
    manifest["laguqa.csv"] = sha256(out / "laguqa.csv")
    manifest["HAK-CIPTA.md"] = sha256(out / "HAK-CIPTA.md")

    (out / "README.md").write_text(card, encoding="utf-8")
    (out / "README.id.md").write_text(card_id, encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{len(manifest) + 1} entri di {out}/")
    print("periksa README.md, terutama bagian Batasan dan Lisensi, "
          "sebelum diunggah.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
