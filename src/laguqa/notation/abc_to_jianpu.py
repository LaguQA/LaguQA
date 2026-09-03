#!/usr/bin/env python3
"""Convert ABC notation back to jianpu (notasi angka) as plain text.

The dataset stores ABC, but the benchmark has to show notasi angka: that is
the notation Indonesian readers actually learned, and it is what the source
songbook prints. A question showing "3 4 5 7 | 1' 7 1'" is worth far more than
the same phrase as "E F G B | c2 B c".

The conversion is deterministic. K: gives the tonic, every note is measured as
an interval from that tonic to get its number, the octave marks come from the
register difference, and the durations come from the ABC note lengths.

It doubles as a cross-check on the transcription: if ABC converted back to
numbers matches the digits printed on the scanned page, that is independent
evidence the transcription is right.

Plain-text convention, documented once here because there is no single
standard for writing jianpu in ASCII:

    1-7     scale degree, 0 is a rest
    #  b    chromatic alteration, written before the digit
    '  ,    one octave up / down, repeated for further octaves
    -       extends the previous note by one beat
    .       extends the previous note by half its value
    _       one beam level: 5_ is an eighth, 5__ a sixteenth
    |       bar line

Usage:
    python -m laguqa.notation.abc_to_jianpu data/abc/006_injit_injit_semut.abc
    python -m laguqa.notation.abc_to_jianpu data/abc/*.abc --check
"""

from __future__ import annotations

import argparse
import re
import sys
from fractions import Fraction
from pathlib import Path

from laguqa.notation.abc_validate import (
    HEADER_RE,
    INLINE_FIELD_RE,
    NOTE_RE,
    TUPLET_RE,
    TUPLET_DEFAULT_Q,
    clean_music,
    parse_key,
    note_to_midi,
)

# Diatonic position of each letter, used to pick between "#4" and "b5": the
# spelling follows the letter the transcriber wrote, not a fixed table.
LETTER_STEP = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
MAJOR_SEMITONES = [0, 2, 4, 5, 7, 9, 11]

# Jianpu counts in beats, and in this songbook the beat is a quarter note.
# L:1/8 therefore makes one ABC unit half a beat.
BEAT_IN_UNITS = Fraction(2)


def tonic_of(text: str) -> tuple[int, str] | None:
    """Return (tonic pitch class, tonic letter) from the K: header."""
    m = re.search(r"^K:(.*)$", text, re.M)
    if not m:
        return None
    pc, _ = parse_key(m.group(1))
    if pc is None:
        return None
    letter = re.match(r"\s*([A-Ga-g])", m.group(1))
    return (pc, letter.group(1).upper()) if letter else None


def degree(midi: int, tonic_pc: int, tonic_letter: str, letter: str) -> str:
    """Render one pitch as a jianpu number with accidental and octave marks.

    The reference octave is the one holding the tonic at octave 4, so for
    "Do = C" the undotted numbers span C4-B4 and 1' is C5. That matches how
    the songbook prints it: on Berkibarlah Benderaku the page leaves 5 (G4)
    undotted and puts a dot above 1 (C5).
    """
    ref = 60 + tonic_pc  # tonic in octave 4; C4 is middle C, MIDI 60
    delta = midi - ref
    octave, semitone = divmod(delta, 12)

    step = (LETTER_STEP[letter.upper()] - LETTER_STEP[tonic_letter]) % 7
    alter = semitone - MAJOR_SEMITONES[step]
    # A letter sitting just below the tonic reads as a large positive semitone
    # distance; pull it back into range so B against C reads as 7, not as a
    # seventh raised by eleven.
    if alter > 6:
        alter -= 12
        octave += 1
    elif alter < -6:
        alter += 12
        octave -= 1

    sign = "#" * alter if alter > 0 else "b" * -alter
    mark = "'" * octave if octave > 0 else "," * -octave
    return f"{sign}{step + 1}{mark}"


def duration(beats: Fraction) -> str:
    """Render a duration as jianpu suffix and trailing extension tokens.

    Returns the suffix glued to the digit plus any separate "-" tokens, so
    "1 - -" (three beats) stays three tokens the way the page prints it.
    """
    if beats <= 0:
        return ""
    whole = int(beats)
    rest = beats - whole

    if whole >= 1:
        suffix = "." if rest == Fraction(1, 2) else ""
        # A remainder the dot cannot express is rare enough to spell out
        # rather than silently round away.
        if rest and rest != Fraction(1, 2):
            suffix = f"[{rest}]"
        return suffix + (" -" * (whole - 1))

    # Below one beat the value is shown by beam levels: each halving adds an
    # underline, and a dot adds half again.
    level = 0
    value = beats
    while value < 1 and level < 6:
        value *= 2
        level += 1
    if value == 1:
        return "_" * level
    if value == Fraction(3, 2):
        return "_" * level + "."
    return f"[{beats}]"


def convert(text: str) -> list[str]:
    """Convert a whole tune. Returns one string per music line."""
    found = tonic_of(text)
    if found is None:
        return []
    tonic_pc, tonic_letter = found

    m = re.search(r"^L:\s*(\d+)/(\d+)", text, re.M)
    unit = Fraction(int(m.group(1)), int(m.group(2))) if m else Fraction(1, 8)
    # Durations are counted in ABC units, then divided by the beat below.
    unit_in_beats = unit / Fraction(1, 8) / BEAT_IN_UNITS

    out: list[str] = []
    in_body = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        h = HEADER_RE.match(line)
        if h and len(h.group(1)) == 1:
            if h.group(1) == "K" and not in_body:
                in_body = True
            continue
        if not in_body:
            continue

        music = INLINE_FIELD_RE.sub(" ", clean_music(line))
        tokens: list[str] = []
        tuplet_left = 0
        close_after = False
        i = 0
        while i < len(music):
            ch = music[i]

            if ch == "(":
                t = TUPLET_RE.match(music, i)
                if t and t.group("p"):
                    p = int(t.group("p"))
                    tuplet_left = int(t.group("r")) if t.group("r") else p
                    tokens.append(f"({p}")
                    i = t.end()
                    continue
                i += 1
                continue

            if ch == "|":
                # Repeat and volta marks are carried through as written; they
                # are part of how the page reads.
                m2 = re.match(r"\|\]|\|\||\|:|:\|\]?|\|", music[i:])
                tokens.append(m2.group(0) if m2 else "|")
                i += len(m2.group(0)) if m2 else 1
                continue

            if ch in ")- \t":
                i += 1
                continue

            nm = NOTE_RE.match(music, i)
            if nm and nm.group("letter"):
                letter = nm.group("letter")
                length = nm.group("length")
                mult = parse_length(length)
                # Inside a tuplet the written value is kept, exactly as the
                # page prints it: three beamed eighths under a "3" bracket.
                # Folding the 2/3 ratio into the duration would produce a
                # third of a beat, which jianpu has no way to write.
                beats = mult * unit_in_beats
                if tuplet_left:
                    tuplet_left -= 1
                    if tuplet_left == 0:
                        close_after = True

                if letter in "zxZ":
                    tokens.append("0" + duration(beats))
                else:
                    acc = nm.group("acc")
                    delta = 0
                    if acc:
                        delta = 0 if acc == "=" else (
                            len(acc) if acc[0] == "^" else -len(acc)
                        )
                    else:
                        _, key_acc = parse_key(re.search(r"^K:(.*)$", text, re.M).group(1))
                        delta = key_acc.get(letter.upper(), 0)
                    midi = note_to_midi(letter, nm.group("octave") or "", delta)
                    tokens.append(degree(midi, tonic_pc, tonic_letter, letter) + duration(beats))
                if close_after:
                    tokens[-1] += ")"
                    close_after = False
                i = nm.end()
                continue
            i += 1

        if tokens:
            out.append(" ".join(tokens))
    return out


def parse_length(raw: str | None) -> Fraction:
    """Interpret the length suffix of an ABC note as a multiple of L:."""
    if not raw:
        return Fraction(1)
    if raw.startswith("/") and raw.count("/") == len(raw):
        return Fraction(1, 2 ** len(raw))
    if "/" in raw:
        num, _, den = raw.partition("/")
        return Fraction(int(num or 1), int(den or 2))
    return Fraction(int(raw))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--check", action="store_true",
                    help="only report files that fail to convert")
    args = ap.parse_args()

    bad = 0
    for name in args.files:
        text = Path(name).read_text(encoding="utf-8")
        lines = convert(text)
        title = re.search(r"^T:(.*)$", text, re.M)
        if not lines:
            print(f"[FAIL] {name}: no music converted")
            bad += 1
            continue
        leftover = sum(ln.count("[") for ln in lines)
        if leftover:
            print(f"[WARN] {name}: {leftover} durations could not be expressed")
            bad += 1
        if args.check:
            continue
        print(f"=== {Path(name).name}  {title.group(1).strip() if title else ''} ===")
        for ln in lines:
            print("  " + ln)
        print()

    if args.check:
        print(f"{len(args.files)} files, {bad} with problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
