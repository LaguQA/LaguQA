#!/usr/bin/env python3
"""Assemble the HuggingFace Space folder from current artefacts.

Everything the Space shows is copied from measured files, never retyped, so the
demo cannot disagree with the thesis. Writes nothing to the Hub; pushing is a
separate, explicit step.

Usage:
    python scripts/23_build_space.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

from laguqa.benchmark.generate import jianpu_bars
from laguqa.benchmark.multichoice import read_mc
from laguqa.paths import BENCHMARK_DIR, CSV_PATH


def build_songs(limit_preview: int) -> list[dict]:
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    out = []
    for r in rows:
        bars = jianpu_bars(r["abc_notation"])
        out.append({
            "id": r["id"], "judul": r["title"], "jenis": r["song_type"],
            "pencipta": r["composer"], "asal": r["origin"],
            "nada_dasar": r["key_signature"], "birama": r["time_signature"],
            "tempo": r["tempo"], "halaman": r["book_page"],
            "abc_status": r["abc_status"],
            # No lyrics column, but not for the reason this comment used to
            # give. The dataset release does publish lyrics, with a per-song
            # rights table in HAK-CIPTA.md. This is a browsing demo: the
            # excerpt and the playable melody are what make a song findable
            # here, and a full lyric column would only make the table wider.
            # Whoever wants the text should get it from the dataset, where it
            # arrives with the rights note attached.
            "notasi": " | ".join(bars[:limit_preview]),
            # Sumber ABC utuh. Notasi baloknya digambar dan dibunyikan abcjs di
            # sisi peramban, jadi yang didengar pengunjung adalah berkas yang
            # sama persis dengan yang menjadi kunci jawaban soal notasi. Versi
            # sebelumnya menyintesis gelombang sinus dari deret MIDI di sini,
            # yang berarti demonya memutar hasil pembacaan ulang, bukan
            # transkripsinya sendiri.
            "abc": r["abc_notation"],
        })
    return out


def build_soal(n: int) -> list[dict]:
    items = read_mc(BENCHMARK_DIR / "laguqa_mc.jsonl")
    return [{"id": x["id"], "kategori": x["kategori"], "tingkat": x["tingkat"],
             "pertanyaan": x["pertanyaan"], "opsi": x["opsi"],
             "kunci": x["kunci"]} for x in items[:n]]


def copy_tables(dest: Path, tabel: Path) -> list[str]:
    """Copy the rendered result tables, and only those.

    Earlier this also shipped every CSV under docs/gambar. The app never read
    them, so they arrived as dead weight that could sit at a different vintage
    from the tables next to them -- a demo that quietly disagrees with itself.
    """
    keluar = dest / "tabel"
    if keluar.is_dir():
        shutil.rmtree(keluar)
    keluar.mkdir(parents=True)
    ada = []
    # Hanya papan skor. Tabel perbandingan di docs/ dibangun terpisah dan
    # sempat tertinggal satu putaran percobaan, sehingga menyalin seluruh *.md
    # berarti menerbitkan angka yang sudah tidak berlaku bersebelahan dengan
    # angka yang berlaku.
    for src in sorted(tabel.rglob("papan-skor.md")):
        target = keluar / f"{src.parent.name}--{src.name}"
        shutil.copy(src, target)
        ada.append(target.name)
    return ada


# --- skor yang bisa dibaca program -------------------------------------------

# The leaderboards name a fine-tuned run "gemma4-e2b [lr4e4]" while the external
# probes name the same run "lr4e4", because there the model column holds the
# adapter and the base row is simply "base". Comparison models carry their own
# name with "-base" on the end. One join key, written once.
def kunci_model(nama: str) -> str:
    # Baris model dasar tidak punya seed di papan skor, jadi namanya tetap
    # polos. Baris adapter punya, dan benchmark luar hanya pernah dijalankan
    # pada seed 1 tiap adapter, sehingga itu yang disasar.
    if nama == "base":
        return "gemma4-e2b"
    if nama.endswith("-base"):
        return nama[:-len("-base")]
    return f"gemma4-e2b [{nama}] seed 1"


# Which prompt condition the Space reports. Both were measured and they agree
# to within 1.5 points; the neutral one is shown because the song-flavoured
# system prompt is the one the adapters were trained under, and reporting it
# would let a label advantage look like knowledge.
PROMPT = "netral"

EKSTERNAL = {"indommlu": "IndoMMLU", "indoculture": "IndoCulture"}


def baca_papan(path: Path) -> tuple[list[str], list[dict]]:
    if not path.is_file():
        return [], []
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    tetap = {"model", "jenis", "seed", "berkas"}
    kolom = [k for k in (rows[0] if rows else {}) if k not in tetap]
    return kolom, rows


def build_scores(tabel: Path) -> dict:
    """Merge both leaderboards and both external probes into one table.

    One row per model, one column per metric, missing cells left missing. The
    Space draws its comparison table and its scatter plot from this and nothing
    else, so a metric it cannot show is a metric that was never measured rather
    than one that was dropped in transit.
    """
    metrik: dict[str, dict] = {}
    model: dict[str, dict] = {}
    lantai: dict[str, float] = {}

    for jalur, awalan, judul in (("mc", "mc", "Pilihan ganda"),
                                 ("full", "full", "Teks bebas")):
        kolom, rows = baca_papan(tabel / jalur / "papan-skor.csv")
        for k in kolom:
            metrik[f"{awalan}.{k}"] = {"label": f"{judul}: {k.lower()}",
                                       "kelompok": judul}
        for r in rows:
            nilai = {f"{awalan}.{k}": float(r[k]) for k in kolom if r[k]}
            if r["jenis"] == "kontrol":
                # The floor is the best a model can score while knowing nothing,
                # so it is the highest control on each metric and not the mean.
                # Only the overall column gets one: on the abstain column the
                # control that never answers scores 100, and calling that a
                # floor would put every real model below it.
                for k, v in nilai.items():
                    if k.endswith(".Keseluruhan"):
                        lantai[k] = max(lantai.get(k, 0.0), v)
            # Controls stay in the table. They are rows a reader compares
            # against, and dropping them would leave the leaderboard without
            # the only entries whose score is known in advance.
            #
            # Seed ikut ke dalam nama begitu satu setelan punya lebih dari satu
            # run. Tanpa itu ketiga run lr4e4 memakai kunci yang sama dan dua di
            # antaranya hilang tertimpa, sehingga tabel menampilkan satu angka
            # seolah hanya itu yang pernah diukur.
            nama = r["model"] + (f" seed {r['seed']}" if r.get("seed") else "")
            m = model.setdefault(nama, {"nama": nama, "jenis": r["jenis"],
                                        "skor": {}})
            m["skor"].update(nilai)

    # Two accounts, two GPUs. Only the base model was measured on both; the rest
    # exist on exactly one, and the first account wins ties. Same weights on
    # different GPUs differ by 0.0-0.4 accuracy points, which the Space says out
    # loud rather than hiding behind a merged column.
    gpu: dict[str, str] = {}
    for folder, nama_gpu in (("akun1-l4", "L4"), ("akun2-l40s", "L40S")):
        path = tabel / "eksternal" / folder / "lupa.csv"
        if not path.is_file():
            continue
        for r in csv.DictReader(path.open(encoding="utf-8")):
            kunci = f"eksternal.{r['dataset']}"
            metrik[kunci] = {"label": EKSTERNAL[r["dataset"]],
                             "kelompok": "Benchmark luar"}
            if r["model"] == "acak":
                lantai.setdefault(kunci, float(r["akurasi"]))
                continue
            if r["prompt"] != PROMPT:
                continue
            nama = kunci_model(r["model"])
            if nama not in model or kunci in model[nama]["skor"]:
                continue
            model[nama]["skor"][kunci] = float(r["akurasi"])
            gpu.setdefault(nama, nama_gpu)

    for nama, g in gpu.items():
        model[nama]["gpu"] = g
    urutan_jenis = {"dilatih": 0, "dasar": 1, "kontrol": 2}
    urut = sorted(model.values(),
                  key=lambda m: (urutan_jenis.get(m["jenis"], 3),
                                 -m["skor"].get("mc.Keseluruhan",
                                                m["skor"].get("full.Keseluruhan", -1))))
    return {"metrik": metrik, "model": urut, "lantai": lantai,
            "prompt_eksternal": PROMPT}


def build_ringkas(songs: list[dict], soal_contoh: int, tabel: Path) -> dict:
    """Counts the header prints, taken from the artefacts rather than typed."""
    mc = read_mc(BENCHMARK_DIR / "laguqa_mc.jsonl")
    papan = tabel / "mc" / "papan-skor.csv"
    pembanding = 0
    if papan.is_file():
        pembanding = sum(1 for r in csv.DictReader(papan.open(encoding="utf-8"))
                         if r["jenis"] == "dasar")
    return {
        "lagu": len(songs),
        "terverifikasi": sum(1 for s in songs
                             if s["abc_status"] == "terverifikasi"),
        "soal": len(mc),
        "kategori": len({x["kategori"] for x in mc}),
        "soal_contoh": soal_contoh,
        "pembanding": pembanding,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("spaces/laguqa-demo"))
    ap.add_argument("--tabel", type=Path, default=Path("docs/tabel"))
    ap.add_argument("--soal", type=int, default=300,
                    help="berapa soal contoh yang ikut ke Space")
    ap.add_argument("--bar-pratinjau", type=int, default=8)
    args = ap.parse_args(argv)

    data = args.out / "data"
    data.mkdir(parents=True, exist_ok=True)

    songs = build_songs(args.bar_pratinjau)
    (data / "lagu.json").write_text(
        json.dumps(songs, ensure_ascii=False), encoding="utf-8")

    soal = build_soal(args.soal)
    (data / "soal.json").write_text(
        json.dumps(soal, ensure_ascii=False), encoding="utf-8")

    tabel = copy_tables(data, args.tabel) if args.tabel.is_dir() else []

    skor = build_scores(args.tabel)
    (data / "skor.json").write_text(
        json.dumps(skor, ensure_ascii=False), encoding="utf-8")

    ringkas = build_ringkas(songs, len(soal), args.tabel)
    (data / "ringkas.json").write_text(
        json.dumps(ringkas, ensure_ascii=False), encoding="utf-8")

    berabc = sum(1 for s in songs if s["abc"].strip())
    print(f"{len(songs)} lagu ({berabc} punya sumber ABC)")
    print(f"{len(soal)} soal contoh dari {ringkas['soal']} soal")
    print(f"{len(tabel)} tabel hasil, {ringkas['pembanding']} model pembanding")
    punya_luar = sum(1 for m in skor["model"]
                     if any(k.startswith("eksternal.") for k in m["skor"]))
    print(f"{len(skor['model'])} model dan {len(skor['metrik'])} metrik di "
          f"skor.json, {punya_luar} di antaranya punya angka benchmark luar")
    print(f"\nditulis ke {args.out}/data/")
    print("berkas app.py, README.md, requirements.txt ditulis terpisah "
          "dan tidak ditimpa perintah ini")
    return 0


if __name__ == "__main__":
    sys.exit(main())
