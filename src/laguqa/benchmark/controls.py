#!/usr/bin/env python3
"""Negative controls: what score does a model get without knowing anything?

An accuracy figure means nothing on its own. If 84 of 107 songs are in 4/4,
then answering "4/4" to every meter question scores 78 percent while knowing
no songs at all. Any trained result has to be read against that floor, not
against zero, and the floor is different for every category.

These controls need no GPU and no model. They fabricate prediction files that
the ordinary scorer reads, so the floor is measured by exactly the code that
measures the models -- not by a separate calculation that could disagree.

    konstan     always the most common answer in that category
    acak        an answer drawn from the category's own answers, seeded
    kosong      empty string, which catches categories where the scorer is
                too generous to notice that nothing was said

"kosong" is the one that finds bugs. A category scoring above zero on empty
predictions has a scoring rule that passes on something other than content.

Usage:
    python scripts/17_controls.py                       # write all three
    python scripts/17_controls.py --regime full
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from laguqa.benchmark.evaluate import REGIME_KEYS
from laguqa.paths import CSV_PATH

SEED = 20260901


def items(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def answers_by_category(rows: list[dict]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for x in rows:
        out[x["kategori"]].append(x["messages"][2]["content"])
    return out


def most_common(pool: dict[str, list[str]]) -> dict[str, str]:
    """The single answer that scores best in each category while knowing nothing.

    Computed from the test side on purpose. That makes it the strongest
    possible constant guesser -- one that has already seen the answer key --
    which is the honest ceiling for a model that has learned no songs. A
    weaker floor computed from the training side would flatter every result.
    """
    return {k: Counter(v).most_common(1)[0][0] for k, v in pool.items()}


def write(rows: list[dict], answer, path: Path, key_path: Path) -> None:
    """Tulis satu berkas kontrol, dengan tajuk versi seperti berkas prediksi.

    Tajuknya sempat hanya ada di jalur pilihan ganda. Akibatnya kontrol teks
    bebas tidak bisa dibedakan dari kontrol versi lama, dan satu-satunya
    pembeda tinggal tanggal berkas -- yang berubah setiap kali disalin.
    Kontrol adalah lantai yang dipakai membaca seluruh angka model, jadi
    kontrol yang basi merusak setiap baris tabel sekaligus.
    """
    sha = hashlib.sha256(key_path.read_bytes()).hexdigest()
    lines = [json.dumps({"berkas_uji": key_path.name, "sha256": sha},
                        ensure_ascii=False)]
    lines += [json.dumps({"id_lagu": x["id_lagu"], "kategori": x["kategori"],
                          "prediksi": answer(x)}, ensure_ascii=False)
              for x in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {path}  ({len(lines) - 1} baris)")


def write_mc(out: Path) -> int:
    """Controls for the multiple-choice track, scored by evaluate_mc."""
    from laguqa.benchmark.multichoice import OUT_PATH, read_mc

    rows = read_mc(OUT_PATH)
    rng = random.Random(SEED)
    huruf = sorted({k for x in rows for k in x["opsi"]})
    sering = Counter(x["kunci"] for x in rows).most_common(1)[0]
    print(f"{len(rows)} soal dari {OUT_PATH.name}\n"
          f"huruf kunci tersering: {sering[0]} "
          f"({sering[1] / len(rows) * 100:.1f}%)\n")

    import hashlib
    sha = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()

    def tulis(nama: str, jawab) -> None:
        lines = [json.dumps({"berkas_uji": OUT_PATH.name, "sha256": sha},
                            ensure_ascii=False)]
        lines += [json.dumps({"id": x["id"], "id_lagu": x["id_lagu"],
                              "kategori": x["kategori"], "prediksi": jawab(x)},
                             ensure_ascii=False) for x in rows]
        p = out / f"kontrol-{nama}--mc.jsonl"
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  {p}  ({len(lines)} baris)")

    # The baseline that matters: knows the book's statistics and not one song.
    # A constant LETTER lands near chance because options are shuffled, which
    # makes every model look informed; the book is 63 percent 4/4 and 70
    # percent Do = C, and a guesser exploiting only that is the real floor.
    #
    # The prior MUST come from the dataset, never from the answer keys. Ranking
    # by "how often is this value a correct answer" scored 82.9 percent on
    # rumpang, whose answers are near-uniform, because a value that is ever an
    # answer scores at least one while a distractor that is never an answer
    # scores zero -- the control was reading the key it was meant to predict.
    # Categories with no corpus-side prior (which song a phrase belongs to)
    # get none, and fall back to chance, which is the honest answer for them.
    kolom = {"birama": "time_signature", "nada_dasar": "key_signature",
             "tempo": "tempo", "asal": "origin", "pencipta": "composer"}
    with open(CSV_PATH, encoding="utf-8") as fh:
        songs = list(csv.DictReader(fh))
    peringkat: dict[str, Counter] = {}
    for kategori, kol in kolom.items():
        peringkat[kategori] = Counter(
            (s[kol] or "").strip() for s in songs if (s[kol] or "").strip())
    # nada_dasar options are written "Do = C" while the column holds "C".
    peringkat["nada_dasar"] = Counter(
        {f"Do = {k}": n for k, n in peringkat["nada_dasar"].items()})
    kata = Counter(w.lower() for s in songs
                   for w in re.findall(r"[\w']+", s["lyrics"] or ""))
    peringkat["rumpang"] = kata

    def prior(x: dict) -> str:
        skor = peringkat.get(x["kategori"])
        if skor is None:
            return rng.choice(sorted(x["opsi"]))
        return max(sorted(x["opsi"]),
                   key=lambda k: skor.get(str(x["opsi"][k]).strip(), 0))

    tulis("konstan", lambda x: sering[0])
    tulis("prior", prior)
    tulis("acak", lambda x: rng.choice(huruf))
    tulis("kosong", lambda x: "")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regime", default="split70",
                    choices=sorted(REGIME_KEYS) + ["mc"])
    ap.add_argument("--out", type=Path, default=Path("hasil"))
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.regime == "mc":
        return write_mc(args.out)

    key_path = REGIME_KEYS[args.regime]
    rows = items(key_path)
    pool = answers_by_category(rows)
    constant = most_common(pool)
    rng = random.Random(SEED)

    print(f"{len(rows)} soal dari {key_path.name}, "
          f"{len(pool)} kategori\n")
    print(f"{'kategori':24} {'n':>5} {'jawaban tersering':<34} {'porsi':>6}")
    for k in sorted(pool):
        vals = pool[k]
        top = constant[k]
        share = vals.count(top) / len(vals) * 100
        print(f"{k:24} {len(vals):>5} {top[:32]:<34} {share:>5.1f}%")

    args.out.mkdir(parents=True, exist_ok=True)
    print()
    write(rows, lambda x: constant[x["kategori"]],
          args.out / f"kontrol-konstan--{args.regime}.jsonl", key_path)
    write(rows, lambda x: rng.choice(pool[x["kategori"]]),
          args.out / f"kontrol-acak--{args.regime}.jsonl", key_path)
    write(rows, lambda x: "",
          args.out / f"kontrol-kosong--{args.regime}.jsonl", key_path)

    print("\nnilai ketiganya dengan scripts/11_evaluate.py. "
          "kontrol-kosong yang di atas nol menandakan aturan penilaian bocor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
