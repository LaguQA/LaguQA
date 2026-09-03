#!/usr/bin/env python3
"""Assemble the multi-variant model repository from the per-run release folders.

WHY THE DEFAULT IS AN ARGUMENT

`PeftModel.from_pretrained(repo)` with no subfolder loads whatever sits at the
root, so the root is the recommendation whether or not anyone says so out loud.
Which run goes there was previously decided by hand-copying files, which left
the decision recorded nowhere and impossible to repeat. Passing it as --bawaan
puts it in the command that built the folder.

WHY THE LICENCE IS FORCED

Every per-run card came out of modal_train.py::release, whose default was
apache-2.0. A LoRA adapter for Gemma is a derivative of Gemma and carries the
Gemma Terms of Use; Apache-2.0 is not a licence its author is free to grant.
The seven cards in the published repository all carried the wrong one. The
frontmatter is rewritten here rather than trusted.

Usage:
    python scripts/29_model_release.py --bawaan gemma4-e2b-full-s1-lr4e4 --apply
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

# Run id to folder name. Two runs are renamed because "penuh" and "penuh15" say
# which training file was used, which means nothing to a reader outside this
# repository, while the epoch count is the thing that actually differs.
NAMA = {
    "gemma4-e2b-full-s1-final": "final",
    "gemma4-e2b-full-s1-lr4e4": "lr4e4",
    "gemma4-e2b-full-s1-lr1e4": "lr1e4",
    "gemma4-e2b-full-s1-r8": "r8",
    "gemma4-e2b-full-s1-r32": "r32",
    "gemma4-e2b-full-s1-penuh": "epoch3",
    "gemma4-e2b-full-s1-penuh15": "epoch3-v15",
}

# Shared by every variant, so they live at the root and nowhere else.
TOKENIZER = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")

# Per-variant, copied into each folder.
ADAPTER = ("adapter_config.json", "adapter_model.safetensors", "manifest.json")

LISENSI = "gemma"


def perbaiki_lisensi(teks: str) -> str:
    return re.sub(r"^license:.*$", f"license: {LISENSI}", teks, count=1,
                  flags=re.M)


def salin(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for nama in ADAPTER:
        shutil.copy2(src / nama, dest / nama)
    # The per-run card becomes CATATAN.md: README.md at the root is the model
    # card for the whole repository, and a second one here would compete with it.
    kartu = src / "README.md"
    if kartu.is_file():
        (dest / "CATATAN.md").write_text(
            perbaiki_lisensi(kartu.read_text(encoding="utf-8")), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("rilis"))
    ap.add_argument("--out", type=Path,
                    default=Path("rilis-model/LaguQA-Gemma4-E2B"))
    ap.add_argument("--bawaan", required=True,
                    help="run id yang ditaruh di akar repositori")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    if args.bawaan not in NAMA:
        raise SystemExit(f"{args.bawaan} bukan run yang dikenal. "
                         f"pilih dari {sorted(NAMA)}")
    ada = [r for r in NAMA if (args.src / r).is_dir()]
    kurang = sorted(set(NAMA) - set(ada))
    if args.bawaan not in ada:
        raise SystemExit(f"tidak ada {args.src / args.bawaan}")

    print(f"bawaan (akar) : {args.bawaan} -> {NAMA[args.bawaan]}")
    for r in sorted(ada):
        if r != args.bawaan:
            print(f"varian        : {r} -> varian/{NAMA[r]}")
    if kurang:
        print(f"tidak ditemukan, dilewati: {kurang}")
    if not args.apply:
        print("\ndry run, tidak ada yang ditulis. tambahkan --apply")
        return 0

    # README.md is the hand-written model card and is not rebuilt here, so it is
    # preserved across a rerun. It does name the default variant, which is why
    # the reminder below is printed rather than left to memory.
    kartu = args.out / "README.md"
    simpan = kartu.read_text(encoding="utf-8") if kartu.is_file() else None
    if args.out.is_dir():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    salin(args.src / args.bawaan, args.out)
    for nama in TOKENIZER:
        shutil.copy2(args.src / args.bawaan / nama, args.out / nama)
    for r in sorted(ada):
        if r != args.bawaan:
            salin(args.src / r, args.out / "varian" / NAMA[r])

    if simpan:
        kartu.write_text(simpan, encoding="utf-8")
    n = sum(1 for p in args.out.rglob("*") if p.is_file())
    mb = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) / 1e6
    print(f"\n{n} berkas, {mb:.0f} MB di {args.out}")
    print(f"lisensi pada tiap CATATAN.md disetel ke \"{LISENSI}\"")
    print("README.md dipertahankan apa adanya. Periksa bahwa bagian "
          "\"Varian mana yang dipakai\" masih cocok dengan bawaan di atas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
