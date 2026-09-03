#!/usr/bin/env python3
"""Repair syllables that a stray capital glued into the middle of a word.

An ABC lyric line marks a word's syllables with hyphens and separates words
with spaces, so "In- do- ne- sia" is one word and "Ha- lo Ha- lo" is two. When
a syllable is transcribed with a capital it should not have, the reassembled
lyric comes out as "IndoNesia", and every benchmark answer built from that
lyric carries the damage.

Thirty-nine such words exist across 24 songs. They are not one problem but
three, and only one of the three can be settled without opening the book:

  the hyphen was right, the capital was wrong   "IndoNesia" -> "Indonesia"
  the capital was right, the hyphen was wrong   "HaloHalo"  -> "Halo Halo"
  both were right                               "karuniaMu", the -Mu suffix

This module fixes the first kind and refuses the second. The distinction is
not a judgement call, it is evidence: a word is only lowercased when that exact
word, in lower case, already appears on its own somewhere in the 107 lyrics.
Ten of the fifteen appear inside the very same song, where the same line was
transcribed correctly one time and not the other.

WHY THE OTHER RULES WERE THROWN OUT

Splitting looked equally automatable and is not. Rules that split when both
halves are known words produced "ber Pancasila" for a word that is simply
"berpancasila", and "Menjun Jung" for "Menjunjung". Indonesian prefixes are
also standalone words, so corpus evidence cannot separate "ke Pemilihan", which
is right, from "ber Padi", which is wrong. Those twenty cases are listed for a
reader holding the book, with the page number for each, and left alone here.

Fixes are written to the .abc lyric lines, not to laguqa.csv, because the csv
column is rebuilt from them by 08_fill_metadata. Fixing the csv alone would be
undone by the next run.

Usage:
    python scripts/14_fix_lyric_joins.py            # check only
    python scripts/14_fix_lyric_joins.py --apply    # rewrite the w: lines
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from laguqa.paths import ABC_DIR, CSV_PATH

csv.field_size_limit(10**8)

MERGED = re.compile(r"[a-z][A-Z]")

# Capitalised by convention when addressing God, and correct as printed.
SUFFIXES = {"mu", "nya", "ku"}


def clean(word: str) -> str:
    return re.sub(r"[^\w'-]", "", word)


def corpus(songs: list[dict]) -> Counter:
    """Every word that appears somewhere without a capital stuck in it.

    This is the dictionary, and it is built from the dataset itself rather than
    from an external word list. That matters because the lyrics are Batak,
    Minang, Sundanese, Ambonese and Papuan as well as Indonesian, and no
    Indonesian dictionary would recognise "tokecang" or "takana".
    """
    seen: Counter = Counter()
    for song in songs:
        for word in song["lyrics"].split():
            token = clean(word)
            if token and not MERGED.search(token):
                seen[token.lower()] += 1
    return seen


def split_at_capital(token: str) -> tuple[str, str]:
    """The word up to the stray capital, and the fragment from it onward."""
    m = MERGED.search(token)
    at = m.start() + 1
    return token[:at], token[at:]


def classify(token: str, words: Counter) -> tuple[str, str]:
    """Return (action, reason). Only "lower" and "keep" are ever acted on."""
    head, tail = split_at_capital(token)
    first = re.match(r"[A-Z][a-z]*", tail)
    fragment = first.group(0) if first else tail

    if fragment.lower() in SUFFIXES:
        return "keep", f"akhiran -{fragment}, memang ditulis begitu"
    if token.lower() in words:
        return "lower", f"{token.lower()!r} sudah ada sebagai kata utuh"
    return "ask", f"{head!r} + {fragment!r}, perlu dilihat di halamannya"


def abc_path(song_id: str) -> Path | None:
    hits = sorted(ABC_DIR.glob(f"{int(song_id):03d}_*.abc"))
    return hits[0] if hits else None


def verses(lines: list[str]) -> list[list[int]]:
    """Line numbers of each verse's w: lines, in singing order.

    A w: line sits under the music line above it, so three w: lines in a row
    are verses one, two and three of the SAME melody, not three consecutive
    lines of one verse. Verse two continues on the next group's second w: line,
    several lines further down the file.

    Getting this wrong is not cosmetic. In song 9 the word broken across a line
    end is "di-" ... "Cium", and the line that starts with "Cium" is four lines
    below the one ending in "di-", with verse two's line in between. Reading
    the file top to bottom puts the wrong line in between and finds nothing.
    """
    groups: list[list[int]] = []
    current: list[int] | None = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("w:"):
            if current is None:
                current = []
                groups.append(current)
            current.append(i)
        elif stripped and not stripped.startswith("%"):
            current = None

    depth = max((len(g) for g in groups), default=0)
    return [[g[k] for g in groups if k < len(g)] for k in range(depth)]


def is_syllable(token: str) -> bool:
    """Bar marks, melismas and skips carry no text and cannot end a word."""
    return any(c.isalpha() for c in token)


def repair(lines: list[str], fragments: set[str]) -> tuple[list[str], list[str]]:
    """Lower-case syllables that a preceding hyphen runs into.

    Only a syllable whose preceding SYLLABLE ends in "-" is touched, skipping
    the "|", "_" and "*" that sit between them: song 63 ends a line with
    "pu- lau pu- |" and starts the next with "Lau", so the bar mark stands
    between the hyphen and the syllable it belongs to.

    A capital that genuinely begins a word follows a syllable with no hyphen,
    and is left exactly as it is.
    """
    parts = {i: line.split() for i, line in enumerate(lines)
             if line.strip().startswith("w:")}
    done: list[str] = []

    for verse in verses(lines):
        stream = [(i, j) for i in verse
                  for j, token in enumerate(parts[i]) if j > 0]
        previous: str | None = None
        for i, j in stream:
            token = parts[i][j]
            if not is_syllable(token):
                continue
            bare = token.rstrip("-").rstrip(",.!?")
            if (previous is not None and previous.endswith("-")
                    and token[:1].isupper()
                    and any(f.lower().startswith(bare.lower()) for f in fragments)):
                parts[i][j] = token[0].lower() + token[1:]
                done.append(f"{token} -> {parts[i][j]}")
            previous = token

    for i, tokens in parts.items():
        lines[i] = " ".join(tokens)
    return lines, done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="tulis ke berkas .abc")
    args = ap.parse_args(argv)

    with open(CSV_PATH, encoding="utf-8") as fh:
        songs = list(csv.DictReader(fh))
    words = corpus(songs)

    buckets: dict[str, list] = defaultdict(list)
    for song in songs:
        for word in song["lyrics"].split():
            token = clean(word)
            if not token or not MERGED.search(token):
                continue
            action, reason = classify(token, words)
            buckets[action].append((song, token, reason))

    print(f"kata menyatu: {sum(len(v) for v in buckets.values())} "
          f"di {len({s['id'] for v in buckets.values() for s, _, _ in v})} lagu\n")

    print(f"diperbaiki otomatis: {len(buckets['lower'])}")
    for song, token, reason in buckets["lower"]:
        _, tail = split_at_capital(token)
        print(f"  lagu {song['id']:>3} hal.{song['book_page']:>4}  "
              f"{token} -> {token[:len(token) - len(tail)] + tail.lower()}   ({reason})")

    print(f"\ndibiarkan, memang benar: {len(buckets['keep'])}")
    for song, token, reason in buckets["keep"]:
        print(f"  lagu {song['id']:>3} hal.{song['book_page']:>4}  {token}   ({reason})")

    print(f"\nPERLU DILIHAT DI BUKU: {len(buckets['ask'])}")
    by_page = sorted(buckets["ask"], key=lambda x: int(x[0]["book_page"]))
    for song, token, reason in by_page:
        print(f"  hal.{song['book_page']:>4}  lagu {song['id']:>3} "
              f"{song['title'][:26]:<28} {token}")

    if not args.apply:
        print("\npratinjau saja. tambahkan --apply untuk menulis ke .abc.")
        return 0

    # Grouped by song, because one file can carry several of these and each
    # rewrite has to see the others' changes rather than overwrite them.
    wanted: dict[str, set[str]] = defaultdict(set)
    for song, token, _ in buckets["lower"]:
        _, tail = split_at_capital(token)
        wanted[song["id"]].add(re.match(r"[A-Z][a-z]*", tail).group(0))

    touched = 0
    for song_id, fragments in sorted(wanted.items(), key=lambda kv: int(kv[0])):
        path = abc_path(song_id)
        if path is None:
            print(f"  lewat: tidak ada .abc untuk lagu {song_id}")
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        lines, done = repair(lines, fragments)
        if not done:
            print(f"  tidak ketemu di .abc: lagu {song_id} {sorted(fragments)}")
            continue
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        touched += 1
        print(f"  lagu {song_id:>3}: {', '.join(done)}")

    print(f"\nberkas .abc diubah: {touched}")
    print("jalankan scripts/08_fill_metadata.py --apply lalu bangkitkan ulang benchmark.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
