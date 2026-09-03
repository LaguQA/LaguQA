#!/usr/bin/env python3
"""Give every composer one spelling, and keep the book's spelling beside it.

WHY THIS MATTERS MORE THAN IT LOOKS

The composer field is not only an answer. It is also where the multiple-choice
builder draws its distractors from, so two spellings of one person become two
options in the same question -- and the question then has two correct answers
while the key names only one. `laguqa-mc-0129` asked who wrote Gugur Bunga and
offered "Ismail Marzuki", "Imail Marzuki", and "Ismail, MZ" as three separate
choices. The key was the third. A model answering with the standard spelling
was marked wrong for being right.

Four questions were broken that way, and in the free-form track three songs
could not be answered correctly at all: their key was a misspelling no model
would reproduce. This is the same defect that substring-sibling options had,
one level up -- there the collision was between strings, here between people.

WHAT IS NOT DONE HERE

Collaborations stay separate entries. "Ibu Sud & Wiratmo S." is not "Ibu Sud",
and merging them would invent a claim the book does not make.

Cornel Simanjuntak (id 60, Maju Tak Gentar) and Alfred Simanjuntak (id 62,
Bangun Pemudi Pemuda) are two different composers with similar names. Only the
typo in the second is repaired. An edit-distance rule would have merged them,
which is why the table below is written out by hand and read by a person.

WHY `NN` BECOMES EMPTY RATHER THAN A NAME

"NN" is nomen nescio, a note that the name is unknown. Left in place it becomes
an answer key, so the benchmark would be asking models to reply "NN" -- testing
knowledge of a cataloguing convention, not of the song. Empty is what the rest
of the dataset already uses for this, and it routes those songs to the abstain
questions, where refusing to name a composer is the correct behaviour.

Usage:
    python scripts/24_fix_composers.py            # pratinjau
    python scripts/24_fix_composers.py --apply
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from laguqa.paths import CSV_PATH

# Written by hand, verified against title and page number, one line per repair.
# Left side is exactly what the book prints; right side is the one spelling the
# dataset will use. An empty right side means the book named nobody.
CANONICAL: dict[str, str] = {
    "Imail Marzuki": "Ismail Marzuki",
    "Ismail, MZ": "Ismail Marzuki",
    "Ismail MZ dan Subroto K.A.": "Ismail Marzuki dan Subroto K.A.",
    "Mukhtar Embut": "Mochtar Embut",
    "Alfred Simanjunatak": "Alfred Simanjuntak",
    "NN": "",
    "NN.": "",
}

PRINTED = "composer_printed"


def rows_of(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, encoding="utf-8", newline="") as fh:
        r = csv.DictReader(fh)
        return list(r.fieldnames or []), list(r)


def plan(rows: list[dict]) -> list[tuple[str, str, str, str]]:
    """(id, title, printed, canonical) for every row this would change."""
    out = []
    for r in rows:
        printed = (r.get("composer") or "").strip()
        if printed in CANONICAL and CANONICAL[printed] != printed:
            out.append((r["id"], r["title"], printed, CANONICAL[printed]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    ap.add_argument("--apply", action="store_true",
                    help="tulis perubahan; tanpa ini hanya pratinjau")
    args = ap.parse_args()

    fields, rows = rows_of(args.csv)
    if PRINTED in fields:
        print(f"kolom {PRINTED} sudah ada, berkas ini sudah dibakukan")
        return 0

    perubahan = plan(rows)
    print(f"{len(rows)} lagu, {len(perubahan)} nama pencipta dibakukan\n")
    for sid, judul, dari, ke in perubahan:
        tujuan = repr(ke) if ke else "kosong (tidak dicantumkan)"
        print(f"  id={sid:>3}  {judul[:32]:32} {dari!r:32} -> {tujuan}")

    sebelum = len({(r.get('composer') or '').strip() for r in rows
                   if (r.get('composer') or '').strip()})
    sesudah = len({CANONICAL.get((r.get('composer') or '').strip(),
                                 (r.get('composer') or '').strip()) for r in rows
                   if CANONICAL.get((r.get('composer') or '').strip(),
                                    (r.get('composer') or '').strip())})
    bernama = sum(1 for r in rows if (r.get("composer") or "").strip())
    bernama_baru = sum(1 for r in rows
                       if CANONICAL.get((r.get("composer") or "").strip(),
                                        (r.get("composer") or "").strip()))
    print(f"\n  pencipta berbeda : {sebelum} -> {sesudah}")
    print(f"  lagu bernama pencipta : {bernama} -> {bernama_baru} "
          f"(tiga lagu ber-'NN' pindah ke soal abstain)")

    if not args.apply:
        print("\npratinjau saja. jalankan lagi dengan --apply untuk menulis.")
        return 0

    # The printed spelling is kept, not overwritten. It is the provenance: a
    # reader checking page 113 must be able to see that the book really does
    # write "Ismail, MZ" there.
    shutil.copy2(args.csv, args.csv.with_suffix(".csv.bak"))
    baru = fields[:]
    baru.insert(fields.index("composer") + 1, PRINTED)
    for r in rows:
        printed = (r.get("composer") or "").strip()
        r[PRINTED] = printed
        r["composer"] = CANONICAL.get(printed, printed)
    with open(args.csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=baru)
        w.writeheader()
        w.writerows(rows)
    print(f"\nditulis {args.csv}, salinan lama di {args.csv.name}.bak")
    print("bangun ulang benchmark dan unggah ulang ke Modal setelah ini.")
    return 0
