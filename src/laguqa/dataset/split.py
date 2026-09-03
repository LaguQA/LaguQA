#!/usr/bin/env python3
"""Split the 107 songs into a train side and a test side, then freeze it.

The split is per song, never per question. If the same song appears on both
sides, the model can answer a test question from a song it was trained on, and
the test score stops meaning anything.

The split must also happen BEFORE augmentation. Augmenting first and then
splitting at random puts variants of the same song on both sides, and that
kind of leak is nearly invisible once it has happened.

Songs are stratified by song_type and abc_status together. Without stratifying,
the test side can end up holding most of the unverified files, which makes its
score incomparable to the train side.

The result is written to data/split.json along with its random seed. Do not
regenerate that file once training has started: every number in the report
refers to it.

Usage:
    python scripts/09_split_songs.py            # preview
    python scripts/09_split_songs.py --apply    # freeze
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from laguqa.paths import CSV_PATH, SPLIT_PATH as OUT_PATH

csv.field_size_limit(10**8)

N_TRAIN = 70
SEED = 20260901


def load_rows() -> list[dict]:
    with open(CSV_PATH, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def split(rows: list[dict], n_train: int, seed: int) -> tuple[list[str], list[str]]:
    """Stratified split. Returns (train ids, test ids), both sorted numerically."""
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in rows:
        strata[(r["song_type"], r["abc_status"])].append(r["id"])

    rng = random.Random(seed)
    train: list[str] = []
    test: list[str] = []
    # Sorting the strata keys keeps the result independent of dict ordering.
    for key in sorted(strata):
        members = sorted(strata[key], key=int)
        rng.shuffle(members)
        n = round(len(members) * n_train / len(rows))
        train.extend(members[:n])
        test.extend(members[n:])

    return sorted(train, key=int), sorted(test, key=int)


def report(rows: list[dict], train: set[str], test: set[str]) -> None:
    by_id = {r["id"]: r for r in rows}
    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for r in rows:
        strata[(r["song_type"], r["abc_status"])].append(r["id"])

    print(f"{'stratum':34} {'total':>7} {'train':>7} {'test':>5} {'% train':>9}")
    for key in sorted(strata):
        ids = strata[key]
        n = sum(1 for i in ids if i in train)
        print(f"{key[0] + ' / ' + key[1]:34} {len(ids):>7} {n:>7} "
              f"{len(ids) - n:>5} {n / len(ids) * 100:>8.1f}%")
    print(f"{'TOTAL':34} {len(rows):>7} {len(train):>7} {len(test):>5} "
          f"{len(train) / len(rows) * 100:>8.1f}%")

    # The test side is printed in full so it can be eyeballed and copied into
    # the report; it is the half that every score will be measured on.
    print(f"\ntest side ({len(test)} songs):")
    for i in sorted(test, key=int):
        r = by_id[i]
        print(f"  {i:>3}  {r['title'][:34]:34} {r['song_type']:9} {r['abc_status']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=N_TRAIN, help="songs on the train side")
    ap.add_argument("--seed", type=int, default=SEED, help="random seed")
    ap.add_argument("--apply", action="store_true", help="write data/split.json")
    args = ap.parse_args()

    rows = load_rows()
    train, test = split(rows, args.train, args.seed)

    print(f"{'APPLY MODE' if args.apply else 'DRY RUN (nothing is written)'}\n")
    report(rows, set(train), set(test))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.apply and OUT_PATH.exists():
        old = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        if old.get("train") != train or old.get("test") != test:
            print(f"\n[STOP] {OUT_PATH.name} already exists and differs.")
            print("A split that has been trained on must not be replaced: every")
            print("experiment number refers to it. Delete the file deliberately")
            print("if you really mean to start over.")
            return 1

    if args.apply:
        OUT_PATH.write_text(
            json.dumps(
                {"seed": args.seed, "n_train": len(train), "n_test": len(test),
                 "train": train, "test": test},
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
        print(f"\nwritten: {OUT_PATH}")
    else:
        print("\nRun with --apply to freeze the split.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
