#!/usr/bin/env python3
"""Rebuild the v1.4 training data that the released root adapter learned from.

WHY THIS EXISTS

`gemma4-e2b-full-s1-lr4e4` is the highest-scoring adapter and sits at the root
of the model repository. Its manifest records train_sha256 f71707cd..., and no
file with that hash survives: `data/benchmark/laguqa_train.jsonl` was rewritten
in place when the benchmark moved to v1.5, and `modal_train.py::upload` pushes
to the data volume with force=True under a fixed name, so both Modal accounts
hold v1.5 as well. Checked, not assumed -- the copies were pulled back down.

The model card therefore said the run could not be reproduced by anyone. That
was wrong, and this script is the disproof.

WHY THE DATA IS RECOVERABLE ANYWAY

v1.4 and v1.5 differ in exactly one input: the composer column. v1.5 was built
after `24_fix_composers.py` gave every composer one spelling and turned the
cataloguing marker "NN" into an empty field. That script does not discard what
it replaces -- it writes the book's own spelling into `composer_printed` beside
it. The normalisation is many-to-one and could not be inverted from the names
alone, but nothing has to be inverted: the original value is still in the
table, one column over.

Restore `composer` from `composer_printed`, and the generator -- same code,
same frozen seed -- emits the v1.4 files byte for byte. Both hashes below are
verified, not copied from a manifest and hoped for.

Only the "all" regime is rebuilt, because that is the one the released adapters
were trained on. The 70/37 split files can be had from the same directory with
`--regime split`, but no published number depends on a v1.4 copy of them.

Nothing in the repository is touched. The rebuilt data lands in its own
directory, reached through LAGUQA_DATA, because v1.5 is the current benchmark
and this is an archive artefact.

Usage:
    python scripts/30_rebuild_v14.py                 # ke data-v14/
    python scripts/30_rebuild_v14.py --keluaran /tmp/v14
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from laguqa.paths import CSV_PATH, DATA_DIR, REPO_ROOT  # noqa: E402

PRINTED = "composer_printed"
GENERATOR = REPO_ROOT / "scripts" / "10_generate_benchmark.py"

# Read off the lr4e4 and penuh manifests, then reproduced by this script.
EXPECTED = {
    "laguqa_train.jsonl":
        "f71707cd4d7951ed344a397044e3d835a631505c1a86034b5656f531ed0a414a",
    "laguqa_test.jsonl":
        "336fa1164955ce2d3e375bf06211c24b8ffe0f288d4aa7e0427072855f0c7ed2",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore_composers(source: Path, target: Path) -> int:
    """Write the pre-normalisation table: composer back to the book's spelling."""
    with open(source, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if PRINTED not in fields:
        raise SystemExit(f"{source} tidak punya kolom {PRINTED}, "
                         "tabel ini belum dibakukan dan tidak perlu dipulihkan")

    changed = 0
    for row in rows:
        if row["composer"] != row[PRINTED]:
            changed += 1
        row["composer"] = row[PRINTED]

    with open(target, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[f for f in fields if f != PRINTED],
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=CSV_PATH,
                    help="tabel lagu yang sudah dibakukan (bawaan: data/laguqa.csv)")
    ap.add_argument("--keluaran", type=Path, default=REPO_ROOT / "data-v14")
    args = ap.parse_args()

    out = args.keluaran.resolve()
    (out / "benchmark").mkdir(parents=True, exist_ok=True)
    changed = restore_composers(args.csv, out / "laguqa.csv")
    print(f"{changed} nama pencipta dikembalikan ke ejaan buku")

    for name in ("split.json", "daftar-isi.csv"):
        shutil.copy2(DATA_DIR / name, out / name)
    abc = out / "abc"
    if not abc.exists():
        abc.symlink_to(DATA_DIR / "abc")

    env = dict(os.environ, LAGUQA_DATA=str(out),
               PYTHONPATH=str(REPO_ROOT / "src"))
    hasil = subprocess.run([sys.executable, str(GENERATOR),
                            "--regime", "all", "--apply"],
                           env=env, capture_output=True, text=True)
    if hasil.returncode != 0:
        sys.stderr.write(hasil.stdout + hasil.stderr)
        return hasil.returncode

    print()
    gagal = False
    for name, expected in EXPECTED.items():
        path = out / "benchmark" / name
        digest = sha256(path)
        cocok = digest == expected
        gagal |= not cocok
        print(f"  {'cocok ' if cocok else 'MELESET'} {name:22} {digest[:16]}")
        # A second copy under a name of its own. The Modal data volume is keyed
        # by file name, so v1.4 and v1.5 can only coexist there if they are not
        # both called laguqa_train.jsonl. modal_train.py picks these up as the
        # "full14" dataset.
        if cocok:
            shutil.copy2(path, path.with_name(path.stem + "_v14" + path.suffix))

    if gagal:
        print("\nHasilnya bukan v1.4. Generator atau tabelnya sudah berubah "
              "lebih jauh daripada kolom pencipta.", file=sys.stderr)
        return 1
    print(f"\nv1.4 tersusun ulang di {out / 'benchmark'}, sama persis dengan "
          "berkas yang dipakai adapter di akar repo model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
