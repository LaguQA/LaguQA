#!/usr/bin/env python3
"""Give every song a name that identifies it, without altering what the book prints.

Two songs in the book are both called "Desaku": id 35 in the regional section
on page 62, and id 89 in the national section on page 142. Different melodies,
different composers, one title. That is fine on paper, where the section a page
belongs to tells them apart, and broken in a benchmark, where a question is
just a sentence.

The damage is not hypothetical. In laguqa_test.jsonl the same sentence appeared
twice with opposing keys:

    "Lagu Desaku diciptakan oleh siapa?"  -> the book does not name a composer
    "Lagu Desaku diciptakan oleh siapa?"  -> L. Manik

No model can be right on both. Whichever it answers, it loses a point, and the
loss says nothing about the model.

The fix keeps two things apart that were being conflated. Column `title` stays
exactly as the book prints it, because that is what a reader compares against
the page. A new column `title_unique` carries a name that identifies exactly
one song, and only the question generators use it. Provenance is untouched;
ambiguity is gone.

Disambiguation follows the book rather than inventing a scheme: the two Desaku
are separated in print by which section they sit in, so they are separated here
by section too, giving "Desaku (Daerah)" and "Desaku (Nasional)".

Nothing is hardcoded. Any title shared by more than one song is handled, so a
future correction that introduces another collision is covered without editing
this file.

Reads and writes data/laguqa.csv. Dry run by default.

Usage:
    python scripts/13_unique_titles.py            # check only
    python scripts/13_unique_titles.py --apply    # check, then write
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict

from laguqa.paths import CSV_PATH

csv.field_size_limit(10**8)

COLUMN = "title_unique"


def normalise(title: str) -> str:
    """Collapse the differences that are only spacing or capitals."""
    return re.sub(r"\s+", " ", title.strip().lower())


def assign(songs: list[dict]) -> tuple[dict[str, str], list[str]]:
    """Map song id to a name that belongs to one song only.

    Returns the mapping and a list of the collisions it had to resolve, so a
    silent change to the dataset cannot slip through as a silent change here.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for song in songs:
        groups[normalise(song["title"])].append(song)

    unique: dict[str, str] = {}
    notes: list[str] = []
    for shared in groups.values():
        if len(shared) == 1:
            unique[shared[0]["id"]] = shared[0]["title"]
            continue

        # The book separates them by section, so that is what separates them
        # here. Page number is the fallback if two songs somehow share both a
        # title and a section, which nothing in the book currently does.
        by_type: dict[str, list[dict]] = defaultdict(list)
        for song in shared:
            by_type[song["song_type"]].append(song)

        for song in shared:
            if len(by_type[song["song_type"]]) == 1:
                unique[song["id"]] = f"{song['title']} ({song['song_type']})"
            else:
                unique[song["id"]] = (f"{song['title']} "
                                      f"(hal. {song.get('book_page', '?')})")
            notes.append(f"lagu {song['id']} {song['title']!r} "
                         f"-> {unique[song['id']]!r}")
    return unique, notes


def check(unique: dict[str, str]) -> list[str]:
    """The one thing that must hold: no two songs may end up with one name."""
    seen: dict[str, str] = {}
    problems: list[str] = []
    for sid, name in sorted(unique.items(), key=lambda kv: int(kv[0])):
        key = normalise(name)
        if key in seen:
            problems.append(f"lagu {sid} dan lagu {seen[key]} sama-sama "
                            f"bernama {name!r} setelah dibedakan")
        seen[key] = sid
    return problems


def write(songs: list[dict], unique: dict[str, str]) -> None:
    fields = list(songs[0].keys())
    if COLUMN not in fields:
        # Placed next to title rather than appended, because a reader scanning
        # the header should meet the two names together.
        fields.insert(fields.index("title") + 1, COLUMN)
    for song in songs:
        song[COLUMN] = unique[song["id"]]

    tmp = CSV_PATH.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(songs)
    tmp.replace(CSV_PATH)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help=f"tulis kolom {COLUMN}")
    args = ap.parse_args(argv)

    with open(CSV_PATH, encoding="utf-8") as fh:
        songs = list(csv.DictReader(fh))

    unique, notes = assign(songs)
    problems = check(unique)

    print(f"lagu: {len(songs)}")
    print(f"judul yang dipakai lebih dari satu lagu: {len(notes) // 2 if notes else 0}")
    for note in notes:
        print(f"  {note}")
    if not notes:
        print("  (tidak ada, kolom akan sama persis dengan title)")

    if problems:
        print(f"\nmasih bentrok setelah dibedakan: {len(problems)}")
        for problem in problems:
            print(f"  {problem}")
        return 1

    changed = sum(1 for s in songs if s.get(COLUMN, "") != unique[s["id"]])
    print(f"\nbaris yang berubah: {changed}")

    if not args.apply:
        print("pratinjau saja. tambahkan --apply untuk menulis.")
        return 0

    write(songs, unique)
    print(f"ditulis: {CSV_PATH.name} (+kolom {COLUMN})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
