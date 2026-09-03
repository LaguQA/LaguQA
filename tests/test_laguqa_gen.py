#!/usr/bin/env python3
"""Checks on the generator's mixture and its train/test separation.

The defect these exist for did not crash anything and did not look wrong. The
training set held 55 "who composed this" examples out of 15,000 while the test
asked 50 of them, weighted equally with a category the training set carried
4,123 times. Every number downstream was then read as evidence about what LoRA
can inject, when it was evidence about what the model had been shown.

Usage:
    python tests/test_laguqa_gen.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter

from laguqa.benchmark.generate import (POOL_TEST, POOL_TRAIN, PARAPHRASE_TRAIN,
                                       askings, sample_to_target)
from laguqa.paths import BENCHMARK_DIR

# No category may take more of the training set than this. Not a target, a
# ceiling: the point is that no single category can crowd the others out again.
# The smallest share of the training set a category may hold. This replaced a
# pair of thresholds -- a 12 percent ceiling on the largest category and a 5:1
# ceiling on the ratio between largest and smallest -- that encoded a rule the
# experiments then disproved.
#
# The rule was equal shares, on the reasoning that the test weights all 21
# categories alike. It fixed what it aimed at: pencipta went from 0.37 percent
# of training to 2.9 and its accuracy from 40 percent to 100. It also cut
# notasi_ke_judul from 4123 examples to 1054, which is a quarter of its
# distinct excerpts, and that category fell from 82 percent to 6.
#
# What actually broke was starvation, not inequality. A category holding more
# content legitimately needs more examples; a category held below about one
# percent is never learned. So the floor is checked and the ceiling is not.
MIN_SHARE = 0.01

# Below this many items a category is limited by the book, not by the budget.
# tempo_abstain exists for two songs in the whole collection.
CONTENT_LIMITED = 20


def load(name: str) -> list[dict]:
    path = BENCHMARK_DIR / name
    if not path.exists():
        return []
    return [json.loads(l) for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()]


def check_mixture() -> int:
    bad = 0
    for name in ("laguqa_train.jsonl", "laguqa_train_split70.jsonl"):
        items = load(name)
        if not items:
            print(f"[LEWAT] {name} belum dibangkitkan")
            continue
        per = Counter(x["kategori"] for x in items)
        kurus = [(k, v) for k, v in sorted(per.items())
                 if v > CONTENT_LIMITED and v / len(items) < MIN_SHARE]
        for k, v in kurus:
            bad += 1
            print(f"[GAGAL] {name}: {k} hanya {v / len(items) * 100:.2f}% "
                  f"data latih ({v} contoh), di bawah batas "
                  f"{MIN_SHARE * 100:.0f}%")
        if not kurus:
            lo = min(v for k, v in per.items() if v > CONTENT_LIMITED)
            print(f"  {name}: kategori terkecil {lo / len(items) * 100:.2f}%, "
                  f"{len(per)} kategori")
    return bad


def check_paraphrase_separation() -> int:
    """A training wording that equals a test wording is a leak, not a variant."""
    bad = 0
    test_wordings = set(POOL_TEST.values())
    for kategori, extras in PARAPHRASE_TRAIN.items():
        for t in extras:
            if t in test_wordings:
                bad += 1
                print(f"[GAGAL] parafrase {kategori} sama dengan sisi uji: {t}")
        if len(set(extras)) != len(extras):
            bad += 1
            print(f"[GAGAL] parafrase {kategori} memuat duplikat")
        if POOL_TRAIN[kategori] in extras:
            bad += 1
            print(f"[GAGAL] parafrase {kategori} mengulang templat latih")

    # And the paraphrases must actually reach the generator.
    if len(askings(POOL_TRAIN, PARAPHRASE_TRAIN, "pencipta")) < 2:
        bad += 1
        print("[GAGAL] askings tidak mengembalikan parafrase untuk sisi latih")
    if len(askings(POOL_TEST, None, "pencipta")) != 1:
        bad += 1
        print("[GAGAL] sisi uji ikut mendapat parafrase")
    return bad


def check_waterfill() -> int:
    """The budget is spent, and no category is asked for more than it has."""
    bad = 0
    rng = random.Random(0)
    items = ([{"kategori": "banyak", "tingkat": "hafalan"}] * 900
             + [{"kategori": "sedang", "tingkat": "hafalan"}] * 60
             + [{"kategori": "sedikit", "tingkat": "hafalan"}] * 5)

    got = sample_to_target(items, 300, rng)
    per = Counter(x["kategori"] for x in got)
    if len(got) != 300:
        bad += 1
        print(f"[GAGAL] anggaran tidak terpakai penuh: {len(got)} dari 300")
    if per["sedikit"] != 5:
        bad += 1
        print(f"[GAGAL] kategori langka tidak diambil seluruhnya: {per['sedikit']}")
    if per["banyak"] > 900 or per["sedang"] > 60:
        bad += 1
        print("[GAGAL] mengambil lebih banyak dari yang tersedia")

    # Asking for more than exists returns everything and does not loop or throw.
    got = sample_to_target(items, 99999, rng)
    if len(got) != len(items):
        bad += 1
        print(f"[GAGAL] permintaan berlebih: {len(got)} dari {len(items)}")
    return bad


def main() -> int:
    bad = check_paraphrase_separation() + check_waterfill() + check_mixture()
    print("lulus" if not bad else f"{bad} gagal")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
