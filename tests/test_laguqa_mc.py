#!/usr/bin/env python3
"""Cases the multiple-choice letter reader has to get right.

Scoring LaguQA-MC is an equality test on a letter, so the only place it can go
wrong is reading that letter out of a free-form reply. Two orderings inside
chosen() were wrong when first written, and both failed in the same direction:
they rejected answers that were correct but not written as a bare letter, which
is how untrained models answer and not how fine-tuned ones do. A scorer with
that bias inflates every trained-versus-baseline gap in the thesis.

Usage:
    python tests/test_laguqa_mc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from laguqa.benchmark.evaluate_mc import chosen
from laguqa.benchmark.multichoice import read_mc

BIRAMA = {"A": "6/8", "B": "2/4", "C": "4/4", "D": "3/4", "E": "5/4"}
NADA_DASAR = {"A": "Do = G", "B": "Do = C", "C": "Do = D", "D": "Do = F",
              "E": "Do = A"}
JUDUL = {"A": "Desaku", "B": "Desaku (hal. 40)", "C": "Halo-Halo Bandung",
         "D": "Gugur Bunga", "E": "Syukur"}
TERTINGGI = {"A": "5", "B": "5'", "C": "3", "D": "1'", "E": "6"}
TEMPO = {"A": "riang", "B": "tempo biasa, riang", "C": "lambat", "D": "cepat",
         "E": "sedang"}

# (reply, options, expected letter, what it checks)
CASES = [
    ("B", BIRAMA, "B", "bare letter"),
    ("B.", BIRAMA, "B", "letter with a full stop"),
    ("**B**", BIRAMA, "B", "markdown emphasis stripped"),
    ("Jawaban: B", BIRAMA, "B", "letter after a label"),
    ("jawaban: e", BIRAMA, "E", "label and letter in lower case"),
    ("B) 2/4", BIRAMA, "B", "letter then the option text"),
    ("Jawabannya adalah B. 2/4", BIRAMA, "B", "letter inside a sentence"),
    ("Pilihan C", BIRAMA, "C", "Indonesian label"),

    # Answering with the option instead of the letter is right in the wrong
    # shape. Rejecting it would score formatting.
    ("2/4", BIRAMA, "B", "option text alone"),
    ("Lagu ini berbirama 4/4.", BIRAMA, "C", "option text inside a sentence"),
    ("Gugur Bunga", JUDUL, "D", "title as the answer"),

    # nada_dasar options end in a letter, so the reply "Do = C" contains a
    # standalone C. Text must beat the bare-letter pass or this returns C.
    ("Do = C", NADA_DASAR, "B", "key name, not the letter C"),
    ("Nada dasarnya Do = G", NADA_DASAR, "A", "key name inside a sentence"),
    ("C", NADA_DASAR, "C", "bare letter still reads as a letter"),

    # An option that is a substring of a sibling. 109 of 1200 items are like
    # this; "one match only" scored the right answer as unreadable.
    ("5'", TERTINGGI, "B", "longer match wins over its own prefix"),
    ("5", TERTINGGI, "A", "the prefix itself still resolves"),
    ("Nada tertingginya 5'", TERTINGGI, "B", "prefix collision in a sentence"),
    ("tempo biasa, riang", TEMPO, "B", "longer tempo phrase wins"),
    ("riang", TEMPO, "A", "shorter tempo phrase resolves"),
    ("Desaku (hal. 40)", JUDUL, "B", "disambiguated title"),

    # Refusing to choose must not be scored as a choice.
    ("Mungkin A atau C", BIRAMA, "", "two letters is a hedge"),
    ("Mungkin 2/4 atau 4/4", BIRAMA, "", "two option texts of equal length"),
    ("Saya tidak tahu.", BIRAMA, "", "no answer given"),
    ("", BIRAMA, "", "empty reply"),
]


def check_cases() -> int:
    bad = 0
    for reply, opsi, want, what in CASES:
        got = chosen(reply, opsi)
        if got != want:
            bad += 1
            print(f"  GAGAL [{what}] {reply!r} -> {got!r}, seharusnya {want!r}")
    return bad


def check_file() -> int:
    """The shipped file must have one key per item and all options distinct."""
    path = Path("data/benchmark/laguqa_mc.jsonl")
    if not path.is_file():
        print(f"  lewat: {path} tidak ada")
        return 0
    bad = 0
    items = read_mc(path)
    for it in items:
        if it["kunci"] not in it["opsi"]:
            print(f"  GAGAL {it['id']}: kunci {it['kunci']} bukan opsi")
            bad += 1
        nilai = [str(v).strip().lower() for v in it["opsi"].values()]
        if len(set(nilai)) != len(nilai):
            print(f"  GAGAL {it['id']}: ada opsi kembar")
            bad += 1
    print(f"  {len(items)} soal diperiksa")
    return bad


def main() -> int:
    bad = check_cases()
    print(f"{len(CASES)} kasus, {bad} gagal")
    file_bad = check_file()
    print(f"pemeriksaan berkas MC: {'lulus' if not file_bad else f'{file_bad} gagal'}")
    return 1 if bad or file_bad else 0


if __name__ == "__main__":
    sys.exit(main())
