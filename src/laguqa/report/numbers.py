#!/usr/bin/env python3
"""Every number the thesis may cite, read from the artefacts rather than typed.

Numbers move when experiments are rerun -- the base model went 19.1 -> 19.8 ->
20.0 through two fixes -- and prose written against an old value goes wrong
silently. So each number is collected here, named, and cited by name.
Each entry keeps the file it came from so it can be traced back.

Usage:
    python scripts/21_numbers.py --json angka.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from laguqa.report.figures import label_of, score_file


class Facts:
    """A name -> (value, source) store that refuses to be quietly overwritten."""

    def __init__(self) -> None:
        self.data: dict[str, dict] = {}

    def add(self, name: str, value, sumber: str, catatan: str = "") -> None:
        if name in self.data and self.data[name]["nilai"] != value:
            # Two computations disagreeing about one name means one of them is
            # wrong, and silently keeping the last is how a thesis ends up
            # quoting a number that no longer matches its own appendix.
            raise SystemExit(
                f"angka '{name}' dihitung dua kali dengan hasil berbeda: "
                f"{self.data[name]['nilai']} (dari {self.data[name]['sumber']}) "
                f"lalu {value} (dari {sumber})")
        self.data[name] = {"nilai": value, "sumber": sumber, "catatan": catatan}

    def __len__(self) -> int:
        return len(self.data)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def dataset_facts(f: Facts, csv_path: Path) -> None:
    if not csv_path.is_file():
        return
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    src = csv_path.name
    f.add("lagu_jumlah", len(rows), src, "baris di berkas dataset")

    jenis = Counter(r["song_type"] for r in rows)
    for nama, n in jenis.items():
        f.add(f"lagu_{nama.lower()}", n, src, "hitungan kolom song_type")

    status = Counter(r["abc_status"] for r in rows)
    for nama, n in status.items():
        f.add(f"abc_{nama.lower()}", n, src, "hitungan kolom abc_status")

    # Which fields the book actually prints. These drive the abstain questions,
    # so the counts are load-bearing rather than decorative.
    for kolom, nama in (("composer", "pencipta"), ("origin", "asal"),
                        ("tempo", "tempo"), ("time_signature", "birama"),
                        ("lyrics", "lirik"), ("abc_notation", "notasi")):
        ada = sum(1 for r in rows if (r.get(kolom) or "").strip()
                  and (r.get(kolom) or "").strip() != "-")
        f.add(f"lagu_ada_{nama}", ada, src, f"kolom {kolom} terisi")
        f.add(f"lagu_tanpa_{nama}", len(rows) - ada, src, f"kolom {kolom} kosong")

    halaman = [int(r["book_page"]) for r in rows if (r.get("book_page") or "").isdigit()]
    if halaman:
        f.add("buku_halaman_awal", min(halaman), src, "book_page terkecil")
        f.add("buku_halaman_akhir", max(halaman), src, "book_page terbesar")


def split_facts(f: Facts, path: Path) -> None:
    if not path.is_file():
        return
    d = json.loads(path.read_text(encoding="utf-8"))
    f.add("split_latih", d["n_train"], path.name, "n_train")
    f.add("split_uji", d["n_test"], path.name, "n_test")
    f.add("split_seed", d["seed"], path.name, "seed yang dibekukan")


def benchmark_facts(f: Facts, manifest: Path, data_dir: Path) -> None:
    if not manifest.is_file():
        return
    d = json.loads(manifest.read_text(encoding="utf-8"))
    src = manifest.name
    for regime, blok in d.items():
        f.add(f"benchmark_versi_{regime}", blok["versi"], src, "versi")
        for berkas, n in blok.get("jumlah", {}).items():
            kunci = "latih" if "train" in berkas else "uji"
            f.add(f"soal_{kunci}_{regime}", n, src, f"jumlah baris {berkas}")
        for berkas, h in blok.get("sha256", {}).items():
            kunci = "latih" if "train" in berkas else "uji"
            f.add(f"sha_{kunci}_{regime}", h, src, f"sha256 {berkas}")

    # Per-category test counts, because the thesis has to say how many
    # questions each reported accuracy rests on.
    uji = data_dir / "laguqa_test.jsonl"
    if uji.is_file():
        c = Counter(json.loads(l)["kategori"]
                    for l in uji.read_text(encoding="utf-8").splitlines()
                    if l.strip())
        f.add("kategori_jumlah", len(c), uji.name, "kategori berbeda di berkas uji")
        for k, n in sorted(c.items()):
            f.add(f"soal_uji_kategori_{k}", n, uji.name, "hitungan kategori")


def run_facts(f: Facts, hasil: Path) -> None:
    for path in sorted(hasil.glob("*-manifest.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        rid = d.get("run_id")
        if not rid:
            continue
        p = f"latih_{rid}"
        for medan, nama in (("gpu", "gpu"), ("epochs", "epoch"),
                            ("total_steps", "langkah"), ("examples", "contoh"),
                            ("seed", "seed"), ("train_runtime_s", "detik"),
                            ("estimated_cost_usd", "usd"),
                            ("peak_memory_gb", "memori_gb"),
                            ("usd_per_hour", "usd_per_jam"),
                            ("train_sha256", "sha_data")):
            if medan in d:
                f.add(f"{p}_{nama}", d[medan], path.name, medan)
        if "train_runtime_s" in d:
            f.add(f"{p}_menit", round(d["train_runtime_s"] / 60, 1),
                  path.name, "train_runtime_s dibagi 60")


def score_facts(f: Facts, hasil: Path, regimes: tuple[str, ...]) -> None:
    for regime in regimes:
        for path in sorted(hasil.glob(f"*--{regime}.jsonl")):
            nama, jenis, benih = label_of(path)
            tally = score_file(path)
            # jenis+benih must be in the name: base and fine-tune share a
            # model name, so omitting them collides the two rows.
            slug = (nama.replace(" ", "_").replace("[", "").replace("]", "")
                    .replace("-", "_"))
            p = f"skor_{regime}_{slug}_{jenis}" + (f"_s{benih}" if benih else "")
            n = sum(v[2] for v in tally.values())
            f.add(f"{p}_soal", n, path.name, "jumlah soal dinilai")
            for i, label in ((0, "tepat"), (1, "toleran")):
                hit = sum(v[i] for v in tally.values())
                f.add(f"{p}_{label}", round(hit / n * 100, 1), path.name,
                      f"akurasi {label} keseluruhan")
            for k, v in sorted(tally.items()):
                f.add(f"{p}_{k}", round(v[1] / v[2] * 100, 1), path.name,
                      f"akurasi toleran kategori {k}, {v[2]} soal")


def as_markdown(f: Facts) -> str:
    lines = ["# Angka terukur", "",
             f"{len(f)} angka, seluruhnya dibaca dari berkas hasil. "
             "Tidak ada yang diketik tangan.", "",
             "| nama | nilai | sumber | cara hitung |", "|---|---|---|---|"]
    for nama in sorted(f.data):
        d = f.data[nama]
        nilai = d["nilai"]
        if isinstance(nilai, str) and len(nilai) > 20:
            nilai = nilai[:16] + "…"
        lines.append(f"| `{nama}` | {nilai} | `{d['sumber']}` | {d['catatan']} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--hasil", type=Path, default=Path("hasil"))
    ap.add_argument("--json", type=Path, help="tulis angka ke berkas JSON")
    ap.add_argument("--markdown", type=Path, help="tulis tabel angka ke berkas")
    args = ap.parse_args(argv)

    f = Facts()
    dataset_facts(f, args.data / "laguqa.csv")
    split_facts(f, args.data / "split.json")
    benchmark_facts(f, args.data / "benchmark" / "laguqa_manifest.json",
                    args.data / "benchmark")
    run_facts(f, args.hasil)
    score_facts(f, args.hasil, ("full", "split70"))

    print(f"{len(f)} angka terkumpul dari {args.data}/ dan {args.hasil}/")
    if args.json:
        args.json.write_text(json.dumps(f.data, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        print(f"ditulis {args.json}")
    if args.markdown:
        args.markdown.write_text(as_markdown(f) + "\n", encoding="utf-8")
        print(f"ditulis {args.markdown}")
    if not args.json and not args.markdown:
        print(as_markdown(f))
    return 0


if __name__ == "__main__":
    sys.exit(main())
