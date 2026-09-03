#!/usr/bin/env python3
"""Re-join syllables that lost their hyphen in the w: lines.

The mirror image of lyric_joins. There a stray capital glued two words into
"IndoNesia"; here a missing hyphen split one word into two, so "bu- leun"
became "bu leun" and the assembled lyric reads "bu leun" where the book prints
"buleun". Every lyric answer built from that line inherits the fault, and a
model spelling the word correctly is marked wrong.

Two forms, both fixed only where the corpus settles them:

    bu leun     a space where a syllable hyphen belongs
    Bu\\-ngong   an escaped hyphen, which ABC renders as a literal "-" inside
                the word rather than as a syllable break

The rule is what keeps this safe: a pair is joined only when that same file
writes it hyphenated elsewhere, so the file contradicts itself. Bungong Jeumpa
has "bu- leun" on one line and "bu leun" on the next. Corpus-wide evidence was
tried first and is far too loose -- one song writing "di- a" for "dia" licensed
joining every "di a" in the book, and "di" is a preposition nearly everywhere.
Pairs with no same-file witness are left alone: "di a" may be one word or two,
and only the page decides.

Dry run by default; pass --apply to rewrite the .abc lyric lines.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

from laguqa.paths import ABC_DIR

TOKEN = re.compile(r"[^\s|]+")


def syllables(line: str) -> list[str]:
    return TOKEN.findall(line[2:] if line.startswith("w:") else line)


# ABC lyric control characters, not syllables. "_" holds the previous syllable
# over another note and "*" skips a note; gluing a syllable to either produces
# a word that exists nowhere.
KONTROL = {"_", "*", "|", "-"}


def witnesses(files: list[Path]) -> Counter:
    """Syllable pairs seen hyphenated: "bu- leun" contributes ("bu", "leun").

    Counted per file and used per file. Corpus-wide is far too loose: one song
    writing "di- a" for "dia" would license joining every "di a" in the book,
    and "di" is a preposition almost everywhere it appears. The evidence that
    actually settles a case is a file contradicting itself, which is what
    Bungong Jeumpa does -- "bu- leun" on one line, "bu leun" on the next.
    """
    seen: Counter = Counter()
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("w:"):
                continue
            toks = syllables(line)
            for a, b in zip(toks, toks[1:]):
                if a.endswith("-") and not a.endswith("\\-"):
                    kiri, kanan = a[:-1], b.rstrip("-")
                    if kiri and kanan and kiri not in KONTROL and kanan not in KONTROL:
                        seen[(kiri.lower(), kanan.lower())] += 1
    return seen


def repair(line: str, seen: Counter) -> tuple[str, list[str]]:
    toks = syllables(line)
    notes: list[str] = []

    # Escaped hyphens first: "Bu\-ngong" is one token that should have been two.
    out: list[str] = []
    for t in toks:
        m = re.match(r"^(.+?)\\-(.+)$", t)
        if m and seen[(m.group(1).lower(), m.group(2).rstrip('-').lower())]:
            out += [m.group(1) + "-", m.group(2)]
            notes.append(f"{t} -> {m.group(1)}- {m.group(2)}")
        else:
            out.append(t)

    joined: list[str] = []
    i = 0
    while i < len(out):
        t = out[i]
        nxt = out[i + 1] if i + 1 < len(out) else ""
        # A capital starts a word, so never join across one. Without this the
        # pass turns "Ha- lo Ha- lo" into "loHa" -- manufacturing exactly the
        # IndoNesia fault that lyric_joins exists to undo.
        if (nxt and not t.endswith("-") and "\\-" not in t
                and t not in KONTROL and nxt.rstrip("-") not in KONTROL
                and not nxt[:1].isupper()
                and seen[(t.lower(), nxt.rstrip("-").lower())]):
            joined.append(t + "-")
            notes.append(f"{t} {nxt} -> {t}- {nxt}")
        else:
            joined.append(t)
        i += 1

    if not notes:
        return line, notes
    # Bar lines carry the syllable-to-note alignment, so they are put back in
    # their original places rather than dropped.
    hasil = line
    for a, b in zip(syllables(line), joined):
        if a != b:
            hasil = re.sub(rf"(?<![^\s|]){re.escape(a)}(?![^\s|])", b, hasil, count=1)
    return hasil, notes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=ABC_DIR)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    files = sorted(args.dir.glob("*.abc"))
    print(f"{len(files)} berkas\n")

    total = 0
    for path in files:
        seen = witnesses([path])
        lines = path.read_text(encoding="utf-8").splitlines()
        baru, catatan = [], []
        for line in lines:
            if line.startswith("w:"):
                hasil, n = repair(line, seen)
                baru.append(hasil)
                catatan += n
            else:
                baru.append(line)
        if catatan:
            total += len(catatan)
            print(f"{path.name}  ({len(catatan)})")
            for c in catatan[:6]:
                print(f"    {c}")
            if args.apply:
                path.write_text("\n".join(baru) + "\n", encoding="utf-8")

    print(f"\n{total} perbaikan"
          + ("" if args.apply else " (dry-run, tambahkan --apply untuk menulis)"))
    if args.apply and total:
        print("jalankan ulang scripts/08_fill_metadata.py agar kolom lirik "
              "dibangun ulang dari w:")
    return 0


if __name__ == "__main__":
    sys.exit(main())
