#!/usr/bin/env python3
"""Package the published dataset as a Kaggle upload.

Built from rilis-dataset/, never from data/ directly, so the Kaggle mirror and
the HuggingFace repository are byte-identical for every file they share. A
mirror assembled by its own rules is not a mirror.

The card is the HuggingFace one with the YAML frontmatter removed: Kaggle has
no use for it and renders it as stray text.

Usage:
    python scripts/28_kaggle_release.py --apply
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

from laguqa.report.dataset_release import DESCRIBE

# Kaggle's licence picker has no plain CC BY-NC 4.0 entry; the nearest listed
# option adds ShareAlike, which is a term this dataset does not carry. "other"
# with the licence named in the card states it correctly instead of asserting a
# condition the author never applied.
LICENSE = "other"
LICENSE_NAMA = "CC BY-NC 4.0"

TAUTAN = {
    "Demo": "https://huggingface.co/spaces/IRedDragonICY/LaguQA-Demo",
    "Model": "https://huggingface.co/IRedDragonICY/LaguQA-Gemma4-E2B",
    "Dataset (HuggingFace)": "https://huggingface.co/datasets/IRedDragonICY/LaguQA",
}


def tanpa_frontmatter(teks: str) -> str:
    if not teks.startswith("---"):
        return teks
    _, _, sisa = teks.partition("---")
    _, _, badan = sisa.partition("---")
    return badan.lstrip("\n")


def build(src: Path, out: Path, slug: str) -> tuple[Path, list[str]]:
    if out.is_dir():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    isi: list[str] = []
    for p in sorted(src.iterdir()):
        if p.name == "README.md":
            continue
        if p.is_dir():
            shutil.copytree(p, out / p.name)
            isi += [f"{p.name}/{q.name}" for q in sorted(p.iterdir())]
        else:
            shutil.copy2(p, out / p.name)
            isi.append(p.name)

    kartu = tanpa_frontmatter((src / "README.md").read_text(encoding="utf-8"))
    (out / "README.md").write_text(kartu, encoding="utf-8")
    isi.append("README.md")

    # Only the files Kaggle shows a description for. Listing the ABC directory
    # entry by entry would put 107 rows in the metadata for one sentence of
    # information.
    resources = [{"path": nama, "description": DESCRIBE[nama]}
                 for nama in sorted(DESCRIBE) if (out / nama).is_file()]
    resources.append({"path": "abc",
                      "description": "transkripsi ABC 2.1, satu berkas satu lagu"})
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "LaguQA",
        "subtitle": ("Benchmark pengetahuan lagu nasional dan daerah Indonesia "
                     "untuk model bahasa"),
        "id": slug,
        "licenses": [{"name": LICENSE}],
        "keywords": ["music", "nlp", "benchmark", "indonesia",
                     "question-answering"],
        "resources": resources,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out, sorted(isi)


def zip_it(folder: Path, target: Path) -> Path:
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(folder))
    return target


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("rilis-dataset"))
    ap.add_argument("--out", type=Path, default=Path("rilis-kaggle/LaguQA"))
    ap.add_argument("--slug", default="ireddragonicy/laguqa")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    if not (args.src / "README.md").is_file():
        raise SystemExit(f"tidak ada {args.src}/README.md. "
                         f"jalankan scripts/18_dataset_release.py --apply dulu")
    if not args.apply:
        print(f"akan membangun {args.out} dari {args.src}, lalu mengemasnya "
              f"jadi zip.\nslug Kaggle: {args.slug}\n"
              f"dry run, tidak ada yang ditulis. tambahkan --apply")
        return 0

    folder, isi = build(args.src, args.out, args.slug)
    zipnya = zip_it(folder, args.out.parent / "LaguQA-kaggle.zip")
    ukuran = zipnya.stat().st_size / 1e6
    sha = hashlib.sha256(zipnya.read_bytes()).hexdigest()

    print(f"{len(isi)} berkas ke {folder}")
    print(f"zip {zipnya}  {ukuran:.1f} MB  sha256 {sha[:16]}")
    print(f"\nlisensi di kartu: {LICENSE_NAMA}. Kaggle tidak punya pilihan itu, "
          f"jadi metadata memakai \"{LICENSE}\";")
    print("pilih \"Other\" pada formulir Kaggle dan biarkan kartunya yang "
          "menyebutkan lisensinya.")
    print("tautan yang tertanam di kartu:")
    for nama, url in TAUTAN.items():
        print(f"  {nama:22} {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
