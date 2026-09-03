#!/usr/bin/env python3
"""The LaguQA leaderboard: every model scored, decomposed by question type.

Shaped like the tables benchmark papers print -- models down the side, an
overall column, then one column per skill -- because that is the shape a reader
already knows how to read. Two departures from that shape, both deliberate.

THE FLOOR IS A ROW, NOT A FOOTNOTE

Benchmark tables invite comparison against zero, and on LaguQA zero is the
wrong reference. The source book is 63 percent 4/4 and 70 percent Do = C, so a
model that answers "4/4" and "Do = C" to everything and knows not one song
scores 33.6 percent overall and 66 percent on key signature. Printing that as
an ordinary row means a reader cannot accidentally skip it, and any model below
the line has been shown to know less than nothing about this book -- which is a
real finding, and the untrained Gemma is one of them.

COLUMNS ARE SKILLS, NOT CATEGORIES

Twenty-one categories is not a table anyone reads. They collapse into six
groups that answer different questions -- recalling a printed fact, admitting
the book is silent, judging a claim, reproducing lyrics, reproducing notation,
and reasoning over notation shown in the prompt. The last group is the only one
a model can do without knowing the book at all, so keeping it separate is what
stops notation reasoning from flattering the memorisation columns. That is not
hypothetical: the first fine-tune here gained nine points overall and every one
of them came from that column.

Every number is recomputed from the prediction files by the same scorer the
rest of the project uses. Nothing here is typed in by hand, so the table cannot
drift away from what was measured.

Usage:
    python scripts/19_leaderboard.py
    python scripts/19_leaderboard.py --regime mc --format latex
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from laguqa.benchmark.evaluate import REGIME_KEYS
from laguqa.report.figures import label_of, score_file

# Question categories grouped into the skills they actually test. A category
# that is not listed still counts towards Keseluruhan; it just gets no column.
GROUPS: dict[str, tuple[str, ...]] = {
    "Fakta": ("pencipta", "asal", "nada_dasar", "tempo", "jenis", "birama"),
    "Abstain": ("pencipta_abstain", "asal_abstain", "tempo_abstain"),
    "Verifikasi": ("verifikasi_pencipta", "verifikasi_asal",
                   "verifikasi_nada_dasar"),
    "Lirik": ("judul_ke_lirik", "judul_ke_baris", "lanjut_lirik", "rumpang",
              "lirik_ke_judul"),
    "Notasi": ("notasi_ke_judul", "judul_ke_notasi"),
    "Penalaran": ("hitung_bar", "nada_tertinggi"),
}

# The MC track asks a different set, so it needs its own columns. Reusing the
# free-form groups would silently print empty cells for categories it has not
# got and drop the two it uniquely has.
GROUPS_MC: dict[str, tuple[str, ...]] = {
    "Fakta": ("pencipta", "asal", "nada_dasar", "tempo", "birama"),
    "Tebak judul": ("pencipta_ke_judul", "asal_ke_judul"),
    "Lirik": ("lirik_ke_judul", "rumpang"),
    "Notasi": ("notasi_ke_judul",),
    "Penalaran": ("hitung_bar", "nada_tertinggi"),
}

# How the rows are banded. Controls first because they set the floor the rest
# is read against, then untrained checkpoints, then fine-tunes.
BANDS = [("Kontrol (tidak mengenal satu lagu pun)", "kontrol"),
         ("Model tanpa pelatihan", "dasar"),
         ("Hasil fine-tuning LaguQA", "dilatih")]

PRETTY = {"konstan": "Tebakan tersering", "acak": "Tebakan acak",
          "kosong": "Tidak menjawab",
          "prior": "Tebakan ikut sebaran kunci"}

# Nama jalur untuk pembaca. `regime` tetap dipakai sebagai nama argumen dan
# nama berkas, tetapi tidak pernah muncul di tabel: kata itu tidak berarti
# apa-apa dalam bahasa Indonesia dan tidak berarti apa-apa juga bagi pembaca
# Inggris yang tidak mengikuti proyek ini.
JALUR = {"mc": "pilihan ganda", "full": "teks bebas",
         "split70": "teks bebas, bagi 70/37"}

# The floor row. On MC the constant-letter guesser lands near chance and would
# put every model above the line; the distribution-aware one is the honest
# reference. Free-form has no prior control, so konstan is its floor.
LANTAI = {"mc": "prior"}


def groups_for(regime: str) -> dict[str, tuple[str, ...]]:
    return GROUPS_MC if regime == "mc" else GROUPS


def tally_mc(path: Path) -> dict[str, tuple[int, int, int]]:
    """category -> (correct, correct, n), shaped like the free-form tally."""
    from laguqa.benchmark.evaluate_mc import score
    from laguqa.benchmark.multichoice import OUT_PATH
    rows, _ = score(path, OUT_PATH)
    per: dict[str, list[int]] = {}
    for r in rows:
        t = per.setdefault(r["kategori"], [0, 0, 0])
        t[0] += r["benar"]
        t[1] += r["benar"]
        t[2] += 1
    return {k: tuple(v) for k, v in per.items()}


def percentages(tally: dict[str, tuple[int, int, int]],
                regime: str) -> dict[str, float]:
    """Overall and per-group lenient accuracy, in percent."""
    out: dict[str, float] = {}
    n_all = sum(v[2] for v in tally.values())
    hit_all = sum(v[1] for v in tally.values())
    out["Keseluruhan"] = hit_all / n_all * 100 if n_all else 0.0
    for name, cats in groups_for(regime).items():
        n = sum(tally[c][2] for c in cats if c in tally)
        hit = sum(tally[c][1] for c in cats if c in tally)
        out[name] = hit / n * 100 if n else float("nan")
    return out


def rows_for(source: Path, regime: str) -> list[dict]:
    files = sorted(source.glob(f"*--{regime}.jsonl"))
    if regime == "mc":
        # One column, one method, and the method is option-text probability --
        # `-peluang-opsi`, not `-peluang` and not generation. All three exist in
        # hasil/ and they disagree by up to 24 points on the same model, so a
        # table mixing them would put one model's format penalty beside
        # another's knowledge and call the difference knowledge. §5.12 records
        # why the other two were tried and dropped. Controls are exempt: they
        # emit a bare letter by construction, so no method applies to them.
        files = [p for p in files
                 if "-peluang-opsi" in p.name or p.name.startswith("kontrol-")]
    if not files:
        raise SystemExit(f"tidak ada prediksi regime {regime} di {source}/")
    out = []
    for path in files:
        name, kind, seed = label_of(path)
        tally = tally_mc(path) if regime == "mc" else score_file(path)
        out.append({"nama": PRETTY.get(name, name), "jenis": kind,
                    "benih": seed, "berkas": path.name,
                    **percentages(tally, regime)})
    return out


def banded(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """Rows split into bands, each sorted by overall score."""
    out = []
    for judul, kind in BANDS:
        band = [r for r in rows if r["jenis"] == kind]
        if band:
            out.append((judul, sorted(band, key=lambda r: -r["Keseluruhan"])))
    return out


def cell(value: float) -> str:
    return "--" if value != value else f"{value:.1f}"


def berkas_kunci(regime: str) -> Path:
    """Berkas soal yang dipakai regime ini. `mc` tidak ada di REGIME_KEYS."""
    if regime == "mc":
        from laguqa.benchmark.multichoice import OUT_PATH
        return OUT_PATH
    return REGIME_KEYS[regime]


def sha_kunci(regime: str) -> str:
    import hashlib
    return hashlib.sha256(berkas_kunci(regime).read_bytes()).hexdigest()


def waktu() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def koma(value: float | None) -> str:
    """One decimal with the Indonesian decimal comma, for prose."""
    return "--" if value is None or value != value else f"{value:.1f}".replace(".", ",")


def kemiringan(kategori: str) -> tuple[str, float]:
    """Most common correct answer in one category, and its share in percent.

    Measured on the questions actually asked, not on the corpus. The two
    differ: 78.5 percent of the 107 songs are in 4/4, but birama questions are
    drawn only from the 57 whose time signature is printed, where the share is
    70.2 percent. The footnote is about what a guesser can exploit, so the
    question file is the right denominator.
    """
    from laguqa.benchmark.multichoice import OUT_PATH
    sub = [d for d in (json.loads(l) for l in
                       OUT_PATH.read_text(encoding="utf-8").splitlines()
                       if l.strip())
           if d.get("kategori") == kategori]
    hitung: dict[str, int] = {}
    for d in sub:
        nilai = str(d["opsi"][d["kunci"]]).strip()
        hitung[nilai] = hitung.get(nilai, 0) + 1
    if not sub:
        return "", float("nan")
    teratas = max(hitung, key=lambda k: hitung[k])
    return teratas, hitung[teratas] / len(sub) * 100


def as_markdown(rows: list[dict], regime: str, floor: float | None) -> str:
    cols = ["Keseluruhan", *groups_for(regime)]
    penilai = "scripts/22_evaluate_mc.py" if regime == "mc" else "scripts/11_evaluate.py"
    ukuran = "akurasi (%)" if regime == "mc" else "akurasi toleran (%)"
    lines = [f"Jalur {JALUR.get(regime, regime)}, {ukuran}. "
             f"Dinilai `{penilai}`.", "",
             "| Model | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for judul, band in banded(rows):
        lines.append(f"| **{judul}** |" + " |" * len(cols))
        for r in band:
            nama = r["nama"] + (f" (seed {r['benih']})" if r["benih"] else "")
            angka = [cell(r[c]) for c in cols]
            if floor is not None and r["jenis"] != "kontrol":
                # Below the floor is the finding, so it is marked rather than
                # left for the reader to notice by comparing two rows.
                angka[0] += " ↓" if r["Keseluruhan"] < floor else ""
            lines.append(f"| {nama} | " + " | ".join(angka) + " |")
    if floor is not None:
        nama_batas = PRETTY[LANTAI.get(regime, "konstan")].lower()
        lines += ["", f"↓ menandai skor di bawah batas bawah, yang dipegang "
                      f"{nama_batas} "
                      f"pada {koma(floor)}%. Model bertanda itu tahu lebih "
                      f"sedikit tentang buku ini daripada penebak yang tidak "
                      f"mengenal satu lagu pun."]
    if "Abstain" in groups_for(regime):
        lines += ["", "Kolom **Abstain** harus dibaca berpasangan dengan "
                      "**Fakta**, tidak sendirian. Soal abstain menguji apakah "
                      "model menolak mengarang ketika bukunya memang tidak "
                      "mencantumkan apa-apa, sehingga model yang menolak "
                      "menjawab segalanya mendapat 100% di sana, dan kedua "
                      "kontrol memang begitu. Angka abstain yang rendah berarti "
                      "model mengarang; angka abstain tinggi berarti sesuatu "
                      "hanya jika kolom Fakta juga tinggi."]
    if regime == "mc":
        # Both figures are read off the table rather than written out, because
        # the controls have been rebuilt twice and a footnote that disagrees
        # with the column above it is worse than no footnote.
        konstan = next((r["Keseluruhan"] for r in rows
                        if r["nama"] == PRETTY["konstan"]), None)
        bir, p_bir = kemiringan("birama")
        nad, p_nad = kemiringan("nada_dasar")
        lines += ["", "Batas bawahnya **penebak yang mengikuti sebaran kunci**, "
                      f"bukan tebakan acak. Kunci soal birama "
                      f"{koma(p_bir)}% bernilai {bir} "
                      f"dan kunci soal nada dasar {koma(p_nad)}% bernilai "
                      f"{nad}, sehingga penebak yang hafal sebaran itu dan nol "
                      f"lagu sudah mendapat {koma(floor)}%. Tebakan huruf "
                      f"tersering hanya {koma(konstan)}% karena opsinya diacak, "
                      "dan memakainya sebagai batas bawah akan membuat setiap model "
                      "tampak berpengetahuan."]
    return "\n".join(lines)


def as_latex(rows: list[dict], regime: str) -> str:
    cols = ["Keseluruhan", *groups_for(regime)]
    head = " & ".join(["\\textbf{Model}"] + [f"\\textbf{{{c}}}" for c in cols])
    lines = ["\\begin{table}[t]", "\\centering", "\\small",
             "\\caption{Akurasi toleran (\\%) pada LaguQA, jalur "
             f"{JALUR.get(regime, regime)}.}}",
             f"\\begin{{tabular}}{{l{'r' * len(cols)}}}",
             "\\toprule", head + " \\\\", "\\midrule"]
    for judul, band in banded(rows):
        lines.append(f"\\multicolumn{{{len(cols) + 1}}}{{c}}{{\\textit{{{judul}}}}} \\\\")
        for r in band:
            nama = r["nama"] + (f" (seed {r['benih']})" if r["benih"] else "")
            lines.append(" & ".join([nama.replace("_", "\\_")]
                                    + [cell(r[c]) for c in cols]) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("hasil"))
    ap.add_argument("--out", type=Path, default=Path("docs/tabel"))
    ap.add_argument("--regime", default="full",
                    choices=sorted(REGIME_KEYS) + ["mc"])
    ap.add_argument("--format", default="semua",
                    choices=("semua", "markdown", "latex", "csv"))
    args = ap.parse_args(argv)

    rows = rows_for(args.dir, args.regime)
    dasar = PRETTY[LANTAI.get(args.regime, "konstan")]
    floor = next((r["Keseluruhan"] for r in rows if r["nama"] == dasar), None)

    out = args.out / args.regime
    out.mkdir(parents=True, exist_ok=True)
    cols = ["Keseluruhan", *groups_for(args.regime)]

    print(as_markdown(rows, args.regime, floor))

    if args.format in ("semua", "markdown"):
        # Stamped with the hash of the question file it describes. A table that
        # survives a benchmark rebuild is indistinguishable from a current one
        # by eye, and this project has already lost an afternoon to exactly
        # that. With the stamp, anyone -- including whoever pastes it into the
        # thesis -- can check it against the file on disk.
        (out / "papan-skor.md").write_text(
            as_markdown(rows, args.regime, floor)
            + f"\n\nDibangun dari {berkas_kunci(args.regime).name} "
              f"sha256 `{sha_kunci(args.regime)[:16]}` pada "
              f"{waktu()}.\n", encoding="utf-8")
    if args.format in ("semua", "latex"):
        (out / "papan-skor.tex").write_text(
            as_latex(rows, args.regime) + "\n", encoding="utf-8")
    if args.format in ("semua", "csv"):
        with open(out / "papan-skor.csv", "w", encoding="utf-8",
                  newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["model", "jenis", "seed", "berkas"] + cols)
            for _, band in banded(rows):
                for r in band:
                    w.writerow([r["nama"], r["jenis"], r["benih"] or "",
                                r["berkas"]] + [round(r[c], 1) for c in cols])

    print(f"\nditulis ke {out}/papan-skor.{{md,tex,csv}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
