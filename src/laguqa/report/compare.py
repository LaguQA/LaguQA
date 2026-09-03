#!/usr/bin/env python3
"""Two or more runs side by side, one row per category, with the difference.

The leaderboard collapses 21 categories into 6 skill columns because that is
what a reader can hold in their head. This is the other table: nothing
collapsed, every category, built for the moment when a change was made on
purpose and the question is which categories it moved. An ablation answered
with a single overall number is not answered -- the first fine-tune here gained
nine points overall and every one of them came from one category out of
twenty-one.

WHY THE DIFFERENCE IS ALSO PRINTED IN QUESTIONS

Most categories hold 50 test questions, so one question is two percentage
points and a category that "improved 6 points" improved by three questions. In
a table of percentages that reads as a result; in questions it reads as what it
is. Both are printed, adjacent, so the second cannot be left out of the
sentence the first invites.

Nothing here is a significance test. Fifty items give a standard error near
seven points for a mid-range accuracy, which means a difference smaller than
about fourteen points between two single runs is not distinguishable from
noise, and the honest response to that is more seeds rather than a p-value
computed as though there were more. SMALL marks the rows where that applies.

Usage:
    python scripts/20_compare.py gemma4-e2b-full-s1 gemma4-e2b-full-s1-seimbang
    python scripts/20_compare.py --regime full a b --out docs/tabel
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from laguqa.benchmark.evaluate import REGIME_KEYS
from laguqa.report.figures import label_of, score_file

# Below this many questions a difference of a couple of items swamps anything
# the change actually did, so the row is marked rather than quietly compared.
NOISE_ITEMS = 7


def resolve_path(source: Path, name: str, regime: str) -> Path:
    """Accept a run id, a filename, or a path. All three get typed in practice."""
    for candidate in (Path(name), source / name,
                      source / f"{name}--{regime}.jsonl"):
        if candidate.is_file():
            return candidate
    raise SystemExit(
        f"tidak menemukan prediksi untuk '{name}'. dicari sebagai berkas, "
        f"dan sebagai {source}/{name}--{regime}.jsonl")


def column(path: Path) -> dict:
    name, kind, seed = label_of(path)
    tally = score_file(path)
    return {"nama": name + (f" (seed {seed})" if seed else ""),
            "jenis": kind, "berkas": path.name, "tally": tally}


def rows(cols: list[dict]) -> list[dict]:
    """One row per category, in descending order of how much the last column
    moved. Sorted that way because the point of the table is to find the
    categories that responded, and alphabetical order buries them."""
    kategori = sorted({k for c in cols for k in c["tally"]})
    out = []
    for k in kategori:
        n = max(c["tally"].get(k, (0, 0, 0))[2] for c in cols)
        hits = [c["tally"].get(k, (0, 0, 0))[1] for c in cols]
        row = {"kategori": k, "soal": n, "benar": hits,
               "persen": [h / n * 100 if n else float("nan") for h in hits]}
        row["selisih_soal"] = hits[-1] - hits[0]
        row["selisih_persen"] = row["persen"][-1] - row["persen"][0]
        out.append(row)
    return sorted(out, key=lambda r: -abs(r["selisih_soal"]))


def total_row(cols: list[dict]) -> dict:
    n = sum(v[2] for v in cols[0]["tally"].values())
    hits = [sum(v[1] for v in c["tally"].values()) for c in cols]
    return {"kategori": "KESELURUHAN", "soal": n, "benar": hits,
            "persen": [h / n * 100 for h in hits],
            "selisih_soal": hits[-1] - hits[0],
            "selisih_persen": (hits[-1] - hits[0]) / n * 100}


def as_markdown(cols: list[dict], regime: str) -> str:
    head = ["Kategori", "Soal"] + [c["nama"] for c in cols] + ["Selisih", "Catatan"]
    from laguqa.report.leaderboard import JALUR
    lines = [f"Jalur {JALUR.get(regime, regime)}, akurasi toleran (%) per "
             f"kategori. Kolom Selisih membandingkan kolom terakhir dengan "
             f"kolom pertama.",
             "",
             "| " + " | ".join(head) + " |",
             "|" + "---|" * len(head)]

    def render(r: dict, tebal: bool = False) -> str:
        nama = f"**{r['kategori']}**" if tebal else r["kategori"]
        angka = [f"{p:.1f}" for p in r["persen"]]
        tanda = "+" if r["selisih_soal"] > 0 else ""
        selisih = (f"{tanda}{r['selisih_persen']:.1f} "
                   f"({tanda}{r['selisih_soal']} soal)")
        catatan = "" if abs(r["selisih_soal"]) >= NOISE_ITEMS else "kecil"
        if r["selisih_soal"] == 0:
            selisih, catatan = "0.0 (0 soal)", "tetap"
        return ("| " + " | ".join([nama, str(r["soal"])] + angka
                                  + [selisih, catatan]) + " |")

    lines.append(render(total_row(cols), tebal=True))
    for r in rows(cols):
        lines.append(render(r))

    lines += ["", f"**kecil** menandai selisih di bawah {NOISE_ITEMS} soal. "
                  f"Pada kategori berisi 50 soal, satu soal bernilai 2 poin "
                  f"persen, sehingga selisih sekecil itu tidak dapat "
                  f"dibedakan dari kebetulan dengan satu seed saja. Tabel ini "
                  f"bukan uji signifikansi; untuk itu diperlukan beberapa "
                  f"seed, bukan perhitungan tambahan atas satu seed."]
    lines += ["", "Berkas yang dibandingkan:"]
    lines += [f"- {c['nama']} — `{c['berkas']}`" for c in cols]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+",
                    help="run id, nama berkas, atau path prediksi")
    ap.add_argument("--dir", type=Path, default=Path("hasil"))
    ap.add_argument("--out", type=Path, default=Path("docs/tabel"))
    ap.add_argument("--regime", default="full", choices=sorted(REGIME_KEYS))
    ap.add_argument("--nama", default="banding",
                    help="nama berkas keluaran tanpa ekstensi")
    args = ap.parse_args(argv)

    if len(args.runs) < 2:
        raise SystemExit("perlu minimal dua run untuk dibandingkan")

    cols = [column(resolve_path(args.dir, r, args.regime)) for r in args.runs]
    teks = as_markdown(cols, args.regime)
    print(teks)

    out = args.out / args.regime
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.nama}.md").write_text(teks + "\n", encoding="utf-8")

    with open(out / f"{args.nama}.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kategori", "soal"]
                   + [f"benar_{c['nama']}" for c in cols]
                   + [f"persen_{c['nama']}" for c in cols]
                   + ["selisih_soal", "selisih_persen"])
        for r in [total_row(cols)] + rows(cols):
            w.writerow([r["kategori"], r["soal"]] + r["benar"]
                       + [round(p, 1) for p in r["persen"]]
                       + [r["selisih_soal"], round(r["selisih_persen"], 1)])

    print(f"\nditulis ke {out}/{args.nama}.{{md,csv}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
