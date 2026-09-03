#!/usr/bin/env python3
"""Score LaguQA-MC: pick the letter the model chose, compare, count.

Scoring is an equality test on a letter, which is the whole reason the
multiple-choice track exists. The only judgement left is reading the letter out
of a free-form reply, and that is where a scorer can quietly become a measure
of formatting instead of knowledge.

An instruction-tuned model told to answer with one letter will still write
"Jawabannya adalah B. 2/4", "**B**", or just "2/4". Accepting only a bare
letter would mark all three wrong and would punish the untrained models far
more than the fine-tuned ones, because terseness is exactly what fine-tuning
teaches. That is the same asymmetry the 256-token ceiling had: a rule that cuts
one side of the comparison and not the other measures verbosity, not knowledge.

So the reply is read in four passes, most explicit first, and anything still
ambiguous is counted as unparseable rather than guessed. The unparseable rate
is reported per model, because if it is high the accuracy below it is not
measuring what it claims to.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from laguqa.benchmark.evaluate import reasoning_of, strip_reasoning
from laguqa.benchmark.multichoice import LETTERS, read_mc

TANDA = re.compile(r"[*_`#]+")
SETELAH_KATA = re.compile(
    r"(?:jawaban(?:nya)?|pilihan(?:nya)?|answer)\s*(?:adalah|ialah|:|=)?\s*"
    r"\(?([A-E])\)?(?![A-Za-z])", re.I)
DI_AWAL = re.compile(r"^\(?([A-E])\)?\s*(?:[.):\-]|$)")
SENDIRIAN = re.compile(r"(?<![A-Za-z])([A-E])(?![A-Za-z])")


def normalise(s: str) -> str:
    return re.sub(r"\s+", " ", TANDA.sub("", s or "")).strip()


def chosen(reply: str, opsi: dict[str, str]) -> str:
    """The letter the reply picks, or "" if it cannot be read unambiguously.

    The reasoning block goes first, for the same reason it does in the
    free-form scorer, except the failure here is quieter and worse. A model
    that works through "bisa A atau B, tapi jawabannya C" names three letters
    and mentions the text of three options, so the bare-letter rule sees an
    ambiguity and gives up while the text rule may lock onto a candidate the
    model had already rejected. Either way the score would measure whether a
    model shows its working, not whether it knows the song -- and five of the
    baselines on this leaderboard are reasoning models.
    """
    teks = normalise(strip_reasoning(reply))
    if not teks:
        return ""

    m = DI_AWAL.match(teks) or SETELAH_KATA.search(teks)
    if m:
        return m.group(1).upper()

    # Naming an option outright is a correct answer written the wrong way, and
    # crediting it is not leniency -- it is refusing to score punctuation. Only
    # when exactly one option matches; two matches means the reply hedged.
    #
    # This runs BEFORE the bare-letter pass because on nada_dasar the two
    # collide: options read "Do = C", "Do = D", so the reply "Do = C" contains
    # a standalone C and the letter pass returns option C, which is a different
    # answer. Text is the more specific evidence, so it wins. The models that
    # answer with the key rather than a letter are the untrained ones, so
    # getting this backwards would have cost the baseline and not the fine-tune.
    # Longest match wins, and a tie is ambiguous. Plain "one match only" looks
    # safer and is not: 109 of 1200 items have an option that is a substring of
    # a sibling -- nada_tertinggi offers both "5" and "5'", tempo offers both
    # "riang" and "tempo biasa, riang". A model answering "5'" matched two
    # options and scored zero for being right in the wrong shape.
    rendah = teks.lower()
    cocok = sorted(((len(str(v).strip()), k) for k, v in opsi.items()
                    if str(v).strip() and str(v).strip().lower() in rendah),
                   reverse=True)
    if cocok and (len(cocok) == 1 or cocok[0][0] > cocok[1][0]):
        return cocok[0][1]

    huruf = {h.upper() for h in SENDIRIAN.findall(teks)}
    huruf &= set(opsi)
    if len(huruf) == 1:
        return huruf.pop()
    return ""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_version(pred_path: Path, mc_path: Path) -> None:
    """Refuse answers written against a different build of the question file.

    Item ids survive a rebuild and so does the count, so without this the
    scorer pairs id to id and reports a perfectly ordinary number for answers
    given to different questions. Controls caught a leak in hitung_bar and the
    rebuilt file kept every id -- that is exactly when this fires.
    """
    head = pred_path.read_text(encoding="utf-8").splitlines()[:1]
    d = json.loads(head[0]) if head else {}
    if "sha256" not in d:
        raise SystemExit(
            f"{pred_path.name} tidak mencantumkan sha256 berkas soal. "
            f"Berkas ini dibuat sebelum penanda versi ada, jadi tidak bisa "
            f"dipastikan menjawab {mc_path.name} yang berlaku. Jalankan ulang "
            f"predict(), atau pindahkan ke hasil/arsip/.")
    ada = sha256(mc_path)
    if d["sha256"] != ada:
        raise SystemExit(
            f"{pred_path.name} menjawab {mc_path.name} versi lain.\n"
            f"  prediksi dibuat atas : {d['sha256'][:16]}\n"
            f"  berkas soal sekarang : {ada[:16]}\n"
            f"Jalankan ulang predict(), atau pindahkan ke hasil/arsip/.")


def score(pred_path: Path, mc_path: Path) -> tuple[list[dict], dict]:
    check_version(pred_path, mc_path)
    items = {x["id"]: x for x in read_mc(mc_path)}
    rows: list[dict] = []
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        item = items.get(p.get("id"))
        if item is None:
            continue
        mentah = (p.get("prediksi") or "").strip()
        pilih = chosen(mentah, item["opsi"])
        rows.append({
            "id": item["id"], "id_lagu": item["id_lagu"],
            "kategori": item["kategori"], "tingkat": item["tingkat"],
            "sumber_kunci": item["sumber_kunci"],
            "kekerasan_pengecoh": item["kekerasan_pengecoh"] or "",
            "pertanyaan": item["pertanyaan"],
            "kunci": item["kunci"], "pilihan": pilih,
            "benar": int(pilih == item["kunci"]),
            "terbaca": int(bool(pilih)),
            # Three columns, not one. The trace is what a reader needs to
            # judge a disputed item, and capping the raw reply at 300
            # characters used to cut it off before the answer even appeared on
            # a reasoning model. `jawaban` is what the scorer actually read.
            "prediksi": mentah,
            "penalaran": reasoning_of(mentah),
            "jawaban": strip_reasoning(mentah),
        })
    if len(rows) != len(items):
        raise SystemExit(
            f"{pred_path.name}: {len(rows)} prediksi cocok dari {len(items)} "
            f"soal di {mc_path.name}. Berkas prediksi ini dibuat atas versi "
            f"benchmark lain, atau tidak memuat medan 'id'.")
    return rows, items


def report(rows: list[dict]) -> None:
    per: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per[r["kategori"]].append(r)

    print(f"\n{'kategori':22} {'n':>5} {'benar':>7} {'terbaca':>8}")
    for k in sorted(per):
        sub = per[k]
        b = sum(x["benar"] for x in sub) / len(sub) * 100
        t = sum(x["terbaca"] for x in sub) / len(sub) * 100
        print(f"{k:22} {len(sub):>5} {b:>6.1f}% {t:>7.1f}%")
    n = len(rows)
    b = sum(x["benar"] for x in rows) / n * 100
    t = sum(x["terbaca"] for x in rows) / n * 100
    print(f"{'JUMLAH':22} {n:>5} {b:>6.1f}% {t:>7.1f}%")

    tak = n - sum(x["terbaca"] for x in rows)
    if tak:
        print(f"\n{tak} jawaban tidak terbaca hurufnya ({tak / n * 100:.1f}%), "
              f"dihitung salah.")
        if tak / n > 0.10:
            print("PERINGATAN: terlalu banyak jawaban tak terbaca. Akurasi di "
                  "atas mengukur kepatuhan format, bukan pengetahuan.")
        contoh = [x["prediksi"][:70] for x in rows if not x["terbaca"]][:3]
        for c in contoh:
            print(f"  contoh: {c!r}")

    # A model that always answers one letter is not answering the questions.
    pilihan = Counter(x["pilihan"] for x in rows if x["terbaca"])
    if pilihan:
        huruf, jml = pilihan.most_common(1)[0]
        bagian = jml / sum(pilihan.values()) * 100
        print(f"\nhuruf paling sering dipilih: {huruf} ({bagian:.1f}%)")
        if bagian > 50:
            print("PERINGATAN: jawaban menumpuk di satu huruf.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prediksi", type=Path)
    ap.add_argument("--mc", type=Path,
                    default=Path("data/benchmark/laguqa_mc.jsonl"))
    ap.add_argument("--tulis-csv", action="store_true")
    args = ap.parse_args(argv)

    rows, _ = score(args.prediksi, args.mc)
    print(f"{args.prediksi.name} dinilai atas {args.mc.name}")
    report(rows)

    if args.tulis_csv:
        out = args.prediksi.with_suffix(".audit.csv")
        with open(out, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nditulis {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
