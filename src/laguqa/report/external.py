#!/usr/bin/env python3
"""Score the external forgetting probes and table the change against the base.

The absolute accuracy is not the finding. Every row is reported as a difference
from the untouched base model measured on the same file, in the same prompt
condition, by the same scorer.

Usage:
    python scripts/27_external_report.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from laguqa.benchmark.evaluate_mc import score
from laguqa.benchmark.multichoice import read_mc
from laguqa.paths import DATA_DIR

KUNCI = {"indommlu": DATA_DIR / "eksternal" / "indommlu_mc.jsonl",
         "indoculture": DATA_DIR / "eksternal" / "indoculture_mc.jsonl"}

JUDUL = {"indommlu": "IndoMMLU", "indoculture": "IndoCulture"}

# Reporting axes per benchmark: the field to break down by, and its heading.
SUMBU = {"indommlu": [("kategori", "grup"), ("subjek", "subjek")],
         "indoculture": [("kategori", "topik"), ("provinsi", "provinsi"),
                         ("tingkat", "cakupan")]}


def label(path: Path, dataset: str) -> tuple[str, str]:
    """Model name and prompt condition, read off the file name."""
    nama = path.name[:-len(f"--{dataset}.jsonl")]
    prompt = "lagu"
    if nama.endswith("-netral"):
        nama, prompt = nama[:-len("-netral")], "netral"
    for buang in ("-peluang-opsi", "-peluang"):
        if nama.endswith(buang):
            nama = nama[:-len(buang)]
    return nama.replace("gemma4-e2b-full-s1-", "").replace("gemma4-e2b-", ""), prompt


def dasar_acak(items: list[dict]) -> float:
    return sum(1 / len(x["opsi"]) for x in items) / len(items)


def kumpulkan(hasil: Path, dataset: str) -> dict[tuple[str, str], list[dict]]:
    kunci = KUNCI[dataset]
    items = {x["id"]: x for x in read_mc(kunci)}
    out: dict[tuple[str, str], list[dict]] = {}
    for pred in sorted(hasil.glob(f"*--{dataset}.jsonl")):
        rows, _ = score(pred, kunci)
        for r in rows:
            # score() carries only the LaguQA columns through; the breakdown
            # fields live on the key file.
            r.update({k: items[r["id"]].get(k, "") for k in
                      ("subjek", "provinsi")})
        out[label(pred, dataset)] = rows
    return out


def akurasi(rows: list[dict]) -> float:
    return sum(r["benar"] for r in rows) / len(rows) * 100 if rows else 0.0


def pembanding(nama: str) -> bool:
    """Model pembanding, bukan salah satu arm percobaan lupa.

    label() memangkas awalan `gemma4-e2b-`, sehingga base model percobaan ini
    bernama persis "base" sedangkan model lain tetap membawa namanya sendiri
    dan berakhiran "-base". Keduanya harus dipisah: seluruh tabel lupa
    melaporkan selisih terhadap base, dan selisih Qwen terhadap base Gemma
    adalah angka yang bentuknya benar tetapi tidak berarti apa-apa.
    """
    return nama.endswith("-base")


def tabel_pembanding(data, prompt) -> list[str]:
    """Akurasi mutlak model pembanding, tanpa selisih.

    Ada di sini bukan untuk mengukur lupa, melainkan untuk menjawab apakah skor
    pada benchmark Indonesia yang sudah mapan memperkirakan skor pada LaguQA.
    Perbandingannya dilakukan di papan skor, bukan di tabel ini.
    """
    baris = [(nama[:-len("-base")], akurasi(rows))
             for (nama, p), rows in data.items()
             if p == prompt and pembanding(nama)]
    if not baris:
        return []
    out = ["", f"Model pembanding, akurasi mutlak, prompt {prompt}.", "",
           "| model | akurasi |", "|---|---:|"]
    out += [f"| {n} | {a:.1f} |" for n, a in sorted(baris, key=lambda x: -x[1])]
    return out


def tabel_utama(data, prompts, acak, n) -> list[str]:
    baris = [f"Tebakan acak {acak:.1%}, {n} soal.", "",
             "| model | prompt | akurasi | selisih dari base |",
             "|---|---|---:|---:|"]
    for prompt in prompts:
        base = data.get(("base", prompt))
        for (nama, p), rows in sorted(data.items()):
            if p != prompt:
                continue
            a = akurasi(rows)
            d = "—" if nama == "base" or base is None else \
                f"{a - akurasi(base):+.1f}"
            baris.append(f"| {nama} | {prompt} | {a:.1f} | {d} |")
    return baris


def tabel_sumbu(data, prompt, medan, judul) -> list[str]:
    base = data.get(("base", prompt))
    if base is None:
        return []
    per_base: dict[str, list[dict]] = defaultdict(list)
    for r in base:
        per_base[r[medan]].append(r)
    model = [n for (n, p) in sorted(data) if p == prompt and n != "base"]
    baris = ["", f"Selisih per {judul}, prompt {prompt}.", "",
             f"| {judul} | n | base | " + " | ".join(model) + " |",
             "|---|---:|---:|" + "---:|" * len(model)]
    for k in sorted(per_base):
        b = akurasi(per_base[k])
        sel = []
        for n in model:
            rows = [r for r in data[(n, prompt)] if r[medan] == k]
            sel.append(f"{akurasi(rows) - b:+.1f}")
        baris.append(f"| {k} | {len(per_base[k])} | {b:.1f} | "
                     + " | ".join(sel) + " |")
    return baris


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hasil", type=Path, default=Path("hasil"))
    ap.add_argument("--out", type=Path, default=Path("docs/tabel/eksternal"))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    teks: list[str] = []
    csv_rows: list[dict] = []
    for dataset, kunci in KUNCI.items():
        if not kunci.is_file():
            print(f"lewat {dataset}: {kunci} tidak ada")
            continue
        data = kumpulkan(args.hasil, dataset)
        if not data:
            print(f"lewat {dataset}: tidak ada berkas prediksi di {args.hasil}")
            continue
        items = read_mc(kunci)
        prompts = sorted({p for (_, p) in data})
        keluarga = {k: v for k, v in data.items() if not pembanding(k[0])}
        teks += [f"## {JUDUL[dataset]}", ""]
        teks += tabel_utama(keluarga, prompts, dasar_acak(items), len(items))
        for prompt in prompts:
            teks += tabel_pembanding(data, prompt)
        for prompt in prompts:
            for medan, judul in SUMBU[dataset]:
                teks += tabel_sumbu(keluarga, prompt, medan, judul)
        teks.append("")
        print(f"{JUDUL[dataset]}: {len(data)} berkas prediksi")
        for (nama, prompt), rows in sorted(data.items()):
            print(f"  {nama:12} {prompt:7} {akurasi(rows):5.1f}%  n={len(rows)}")
            csv_rows.append({"dataset": dataset, "model": nama,
                             "prompt": prompt, "akurasi": f"{akurasi(rows):.1f}",
                             "n": len(rows)})
        # The random-guess floor travels with the numbers. Both files have five
        # options on most items and fewer on some, so it is neither 20 nor 25
        # and cannot be reconstructed by whoever reads the CSV later.
        csv_rows.append({"dataset": dataset, "model": "acak", "prompt": "",
                         "akurasi": f"{dasar_acak(items) * 100:.1f}",
                         "n": len(items)})

    if not teks:
        return 0
    print()
    print("\n".join(teks))
    if not args.apply:
        print("\ndry run, tidak ada yang ditulis. tambahkan --apply")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "lupa.md").write_text("\n".join(teks) + "\n", encoding="utf-8")
    # The same numbers in a form other programs can read. The Space plots them
    # beside the LaguQA scores, and parsing them back out of the markdown would
    # mean two copies that can disagree.
    with open(args.out / "lupa.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "model", "prompt",
                                           "akurasi", "n"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\nditulis {args.out / 'lupa.md'} dan {args.out / 'lupa.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
