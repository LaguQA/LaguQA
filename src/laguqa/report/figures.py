#!/usr/bin/env python3
"""Build the figures and tables the thesis prints, from finished runs.

Reads what fetch() brought down into hasil/ -- one manifest per run, one
prediction file per evaluated model -- and writes vector figures plus the CSV
each figure was drawn from. The CSV matters as much as the figure: an examiner
who wants to check a bar can read the number, and a figure that has drifted
from its data is caught by regenerating rather than by trusting.

Scoring is not re-implemented here. It imports score() from benchmark.evaluate,
the same function the tests pin, so a figure cannot disagree with the scorer.

Three figures, each answering one question:

    loss        did it learn                    one line per run
    kategori    what did it learn               baseline against trained
    model       which base model learned most   three seeds, with error bars

The error bars are the reason three seeds exist. A single run's accuracy is one
draw; without a spread there is no way to say whether a two-point gap between
two models means anything. Bars are the range across seeds, not a standard
error, because three points do not justify assuming a distribution.

Usage:
    python scripts/16_figures.py                    # all figures found
    python scripts/16_figures.py --dir hasil --out docs/gambar
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display on a headless machine, and none is needed
import matplotlib.pyplot as plt  # noqa: E402

from laguqa.benchmark.evaluate import (  # noqa: E402
    REGIME_KEYS, check_version, load_keys, rival_sets, rows_of, score,
)

# Printed at one column of an A4 page with 3 cm margins, so text set at 9 pt
# here lands at 9 pt on paper. Figures scaled by the word processor come out
# with labels smaller than the body text.
plt.rcParams.update({
    "figure.figsize": (6.3, 3.6),
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

# Greys and one accent, because the thesis prints in black and white. A figure
# that only reads in colour becomes unreadable at exactly the moment it matters.
BASE_COLOUR = "#B0B0B0"
TRAINED_COLOUR = "#1F1F1F"
SEED_MARKERS = ("o", "s", "^")


def write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {path}")


def manifests(source: Path) -> dict[str, dict]:
    """Every run manifest in the results directory, keyed by run id."""
    out: dict[str, dict] = {}
    for path in sorted(source.glob("*-manifest.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("tag") == "smoke":
            continue  # thirty steps, not a result
        out[data["run_id"]] = data
    return out


def score_file(path: Path) -> dict[str, tuple[int, int, int]]:
    """Score one prediction file. Returns category -> (strict, lenient, n).

    Raises if the file does not line up with the current key file. A prediction
    made against an older benchmark still parses, still carries valid song ids,
    and still scores -- on whichever subset happens to survive. A stale pilot
    file scored 26.2 percent here on 724 of 766 questions and would have gone
    into a chart beside a current run as though the two were comparable.

    A chart is the wrong place to be lenient about that: figures get read
    without the log that produced them, so this stops rather than warns.
    """
    regime = path.stem.rsplit("--", 1)[-1]
    key_path = REGIME_KEYS.get(regime, REGIME_KEYS["full"])
    check_version(path, key_path)
    keys = load_keys(key_path)
    rivals = rival_sets(keys)
    used: dict[tuple[str, str], int] = defaultdict(int)
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    missing = 0

    for p in rows_of(path):
        slot = (p["id_lagu"], p["kategori"])
        bucket = keys.get(slot)
        if not bucket:
            missing += 1
            continue
        gold = bucket[min(used[slot], len(bucket) - 1)]["messages"][2]["content"]
        used[slot] += 1
        s, l = score(p["kategori"], p.get("prediksi", ""), gold,
                     rivals[p["kategori"]])
        t = tally[p["kategori"]]
        t[0] += s
        t[1] += l
        t[2] += 1

    n = sum(v[2] for v in tally.values())
    if missing or n != len(keys_total(keys)):
        raise SystemExit(
            f"{path.name} tidak cocok dengan kunci {regime} yang berlaku: "
            f"{n} soal ternilai, {missing} tanpa kunci, "
            f"{len(keys_total(keys))} soal di berkas kunci.\n"
            f"Berkas prediksi ini dibuat atas benchmark versi lain. "
            f"Jalankan ulang predict(), atau pindahkan berkasnya ke "
            f"hasil/arsip/ supaya tidak ikut tergambar.")
    return {k: tuple(v) for k, v in tally.items()}


def keys_total(keys: dict) -> list:
    return [x for items in keys.values() for x in items]


def label_of(path: Path) -> tuple[str, str, str | None]:
    """(model, kind, seed) read off a prediction filename.

    predict() writes "<run_id>--<regime>.jsonl", and run_id is either
    "<model>-base" for an untrained checkpoint or "<model>-<regime>-s<seed>"
    for a trained one, optionally followed by "-<tag>". The filename is the
    only record tying a prediction to what produced it, so it is parsed rather
    than guessed at.

    The tag has to reach the label. Two arms of an ablation differ by nothing
    else -- same model, same seed -- so dropping it would draw them as one bar
    labelled twice, which is worse than not drawing them at all.
    """
    stem = path.stem.rsplit("--", 1)[0]
    # The scoring method is part of the filename, not part of the model name.
    # Left in place it defeats every branch below: "lfm25-2b-base-peluang" does
    # not end in "-base", so an untrained model would be filed as a trained one
    # and drawn in the wrong band. Longest suffix first, since "-peluang" is a
    # prefix of "-peluang-opsi".
    for metode in ("-peluang-opsi", "-peluang"):
        if stem.endswith(metode):
            stem = stem[: -len(metode)]
            break
    if stem.startswith("kontrol-"):
        # Not a model at all. Left out of the model bars and drawn as a floor
        # line instead, because a control charted as though it were a trained
        # system invites exactly the misreading the controls exist to prevent.
        return stem[len("kontrol-"):], "kontrol", None
    if stem.endswith("-base"):
        return stem[:-5], "dasar", None
    # full14 ikut dikenali. Ia rezim yang sama dengan full, hanya berkas
    # latihnya versi 1.4; tanpa nama ini di pola, run seed 2 dan 3 milik lr4e4
    # tidak terbaca sama sekali dan papan skor menampilkan satu seed seolah
    # itulah seluruh buktinya.
    m = re.match(r"(.+?)-(?:split70|full14|full)-s(\d+)(?:-(.+))?$", stem)
    if m:
        name = m.group(1) + (f" [{m.group(3)}]" if m.group(3) else "")
        return name, "dilatih", m.group(2)
    return stem, "dilatih", None


# --- figure one: did it learn ------------------------------------------------


def figure_loss(runs: dict[str, dict], out: Path) -> None:
    """Training loss as a line, validation loss as points on the same axes.

    Both together, because either alone answers the wrong question. Training
    loss always falls; it says the optimiser works, not that the model is
    learning anything transferable. Validation loss turning back upward while
    training loss keeps dropping is what overfitting looks like, and it is the
    only evidence in this project that can say whether three epochs was one
    epoch too many.
    """
    if not runs:
        return
    fig, ax = plt.subplots()
    rows: list[tuple] = []
    for i, (run_id, m) in enumerate(sorted(runs.items())):
        history = [h for h in m.get("loss_history", []) if "loss" in h]
        if not history:
            continue
        steps = [h["step"] for h in history]
        loss = [h["loss"] for h in history]
        line, = ax.plot(steps, loss, linewidth=1.2, label=f"{run_id} (latih)")
        rows += [(run_id, "latih", s, v) for s, v in zip(steps, loss)]

        evals = m.get("eval_loss_history", [])
        if evals:
            ax.plot([h["step"] for h in evals], [h["eval_loss"] for h in evals],
                    linestyle="--", linewidth=1.0,
                    marker=SEED_MARKERS[i % len(SEED_MARKERS)], markersize=4,
                    color=line.get_color(), label=f"{run_id} (validasi)")
            rows += [(run_id, "validasi", h["step"], h["eval_loss"])
                     for h in evals]

    ax.set_xlabel("langkah")
    ax.set_ylabel("loss")
    ax.set_yscale("log")  # the first hundred steps drop further than the rest
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(out / "loss.pdf")
    fig.savefig(out / "loss.png")
    plt.close(fig)
    print(f"  {out / 'loss.pdf'}")
    write_csv(out / "loss.csv", ["run_id", "jenis", "step", "loss"], rows)


# --- figure two: what did it learn -------------------------------------------


def figure_categories(scored: dict[Path, dict], out: Path) -> None:
    """Baseline against trained, category by category, one model at a time."""
    by_model: dict[str, dict[str, dict]] = defaultdict(dict)
    constant: dict = {}
    for path, tally in scored.items():
        model, kind, seed = label_of(path)
        if kind == "kontrol":
            if model == "konstan":
                constant = tally
            continue
        if kind == "dasar" or seed == "1":
            by_model[model][kind] = tally

    for model, sides in sorted(by_model.items()):
        if len(sides) < 2:
            continue
        cats = sorted(set(sides["dasar"]) | set(sides["dilatih"]))
        fig, ax = plt.subplots(figsize=(6.3, 0.32 * len(cats) + 1.4))
        y = range(len(cats))
        rows: list[tuple] = []

        def pct(side: str, c: str) -> float:
            s, l, n = sides[side].get(c, (0, 0, 0))
            return l / n * 100 if n else 0.0

        ax.barh([i + 0.2 for i in y], [pct("dasar", c) for c in cats],
                height=0.38, color=BASE_COLOUR, label="sebelum dilatih")
        ax.barh([i - 0.2 for i in y], [pct("dilatih", c) for c in cats],
                height=0.38, color=TRAINED_COLOUR, label="sesudah dilatih")

        # One tick per category marking what the constant guesser scores there.
        # Without it this chart lies by omission: nada_dasar reads as 0 -> 70
        # percent, a triumph, when 70.3 percent is simply how often "Do = C"
        # is the right answer and the model said "Do = C" every single time.
        if constant:
            for i, c in enumerate(cats):
                s, l, n = constant.get(c, (0, 0, 0))
                if not n:
                    continue
                ax.plot([l / n * 100], [i], marker="|", markersize=11,
                        markeredgewidth=1.6, color="#C0392B", linestyle="none",
                        label="tebakan konstan" if i == 0 else None)

        for c in cats:
            s, l, n = sides["dilatih"].get(c, (0, 0, 0))
            bs, bl, bn = sides["dasar"].get(c, (0, 0, 0))
            ks, kl, kn = constant.get(c, (0, 0, 0))
            rows.append((model, c, bn, round(bl / bn * 100, 1) if bn else 0,
                         round(kl / kn * 100, 1) if kn else 0,
                         n, round(l / n * 100, 1) if n else 0,
                         round(s / n * 100, 1) if n else 0))

        ax.set_yticks(list(y))
        ax.set_yticklabels(cats, fontsize=8)
        ax.set_xlabel("jawaban benar (%), penilaian toleran")
        ax.set_xlim(0, 100)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.set_title(model, fontsize=9, loc="left")
        fig.savefig(out / f"kategori-{model}.pdf")
        fig.savefig(out / f"kategori-{model}.png")
        plt.close(fig)
        print(f"  {out / f'kategori-{model}.pdf'}")
        write_csv(out / f"kategori-{model}.csv",
                  ["model", "kategori", "n_dasar", "dasar_toleran",
                   "konstan_toleran", "n_dilatih", "dilatih_toleran",
                   "dilatih_tepat"], rows)


# --- figure three: which base model learned most -----------------------------


def figure_models(scored: dict[Path, dict], out: Path) -> None:
    """One bar per model, whiskers showing the spread across seeds."""
    points: dict[tuple[str, str], list[float]] = defaultdict(list)
    floors: dict[str, float] = {}
    for path, tally in scored.items():
        model, kind, _ = label_of(path)
        total_l = sum(v[1] for v in tally.values())
        total_n = sum(v[2] for v in tally.values())
        if not total_n:
            continue
        if kind == "kontrol":
            floors[model] = total_l / total_n * 100
        else:
            points[(model, kind)].append(total_l / total_n * 100)

    models = sorted({m for m, _ in points})
    if not models:
        return
    fig, ax = plt.subplots()
    rows: list[tuple] = []
    width = 0.38
    for i, model in enumerate(models):
        for offset, kind, colour in ((-width / 2, "dasar", BASE_COLOUR),
                                     (width / 2, "dilatih", TRAINED_COLOUR)):
            vals = points.get((model, kind), [])
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            lo, hi = min(vals), max(vals)
            ax.bar(i + offset, mean, width=width, color=colour,
                   label=kind if i == 0 else None)
            if len(vals) > 1:
                # Range, not standard error: three seeds do not justify
                # assuming a distribution, and the range is what was observed.
                ax.errorbar(i + offset, mean, yerr=[[mean - lo], [hi - mean]],
                            fmt="none", ecolor="#404040", capsize=3, linewidth=1)
            for v in vals:
                ax.plot(i + offset, v, "o", color="#707070", markersize=2.5)
            rows.append((model, kind, len(vals), round(mean, 2),
                         round(lo, 2), round(hi, 2)))

    # The constant guesser is the number every bar has to clear to mean
    # anything, so it is drawn across the whole chart rather than tucked into
    # a caption. Without it a 40 percent bar looks like a result.
    for name, value in sorted(floors.items()):
        if name == "kosong":
            continue  # a zero line adds nothing but ink
        ax.axhline(value, linestyle=":", linewidth=1, color="#606060")
        # Axes fraction for x, data units for y: the label then sits just
        # inside the right edge whatever the bar count, instead of landing
        # outside the axes as it did when x was computed from len(models).
        ax.annotate(f"kontrol {name}: {value:.1f}%",
                    xy=(0.995, value), xycoords=ax.get_yaxis_transform(),
                    fontsize=7, va="bottom", ha="right", color="#404040")
        rows.append((f"kontrol-{name}", "kontrol", 1, round(value, 2),
                     round(value, 2), round(value, 2)))

    # Keeps one or two models from being drawn as absurdly wide bars.
    ax.set_xlim(-0.6, len(models) - 0.4)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=8)
    ax.set_ylabel("jawaban benar (%), penilaian toleran")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out / "model.pdf")
    fig.savefig(out / "model.png")
    plt.close(fig)
    print(f"  {out / 'model.pdf'}")
    write_csv(out / "model.csv",
              ["model", "kondisi", "n_seed", "rerata", "terendah", "tertinggi"],
              rows)


def table_decomposition(scored: dict[Path, dict], out: Path) -> None:
    """Per category: untrained, constant guesser, trained, and the difference.

    The most important table in the project, because the headline accuracy is
    not interpretable without it. The first trained run scored 36.6 percent
    against a 27.5 percent floor, which reads like a nine-point gain until the
    rows are separated: the entire gain sat in bar counting and highest-note
    reading, while nada_dasar matched the constant guesser exactly -- the model
    answered "Do = C" to all 37 questions and scored the base rate for it.

    A single number cannot tell those two things apart. This table can, so it
    is written whenever a trained run and a constant control are both present.
    """
    def whole(tally: dict) -> tuple[float, int]:
        l = sum(v[1] for v in tally.values())
        n = sum(v[2] for v in tally.values())
        return (l / n * 100 if n else 0.0), n

    sides: dict[str, dict] = {}
    for path, tally in scored.items():
        name, kind, seed = label_of(path)
        if kind == "kontrol" and name == "konstan":
            sides["konstan"] = tally
        elif kind == "dasar":
            sides.setdefault("dasar", tally)
        elif kind == "dilatih" and seed in (None, "1"):
            sides.setdefault("dilatih", tally)

    if not {"konstan", "dilatih"} <= set(sides):
        return

    def pct(side: str, c: str) -> float:
        s, l, n = sides.get(side, {}).get(c, (0, 0, 0))
        return l / n * 100 if n else 0.0

    rows = []
    for c in sorted(sides["dilatih"]):
        n = sides["dilatih"][c][2]
        d, k, t = pct("dasar", c), pct("konstan", c), pct("dilatih", c)
        # "semu" marks a score that matches the constant guesser: the model
        # scored it by repeating the majority answer, not by knowing anything.
        verdict = ("semu" if abs(t - k) < 1.0
                   else "naik" if t - k > 10
                   else "turun" if t - k < -5
                   else "")
        rows.append((c, n, round(d, 1), round(k, 1), round(t, 1),
                     round(t - k, 1), verdict))

    for side in ("dasar", "konstan", "dilatih"):
        if side in sides:
            value, n = whole(sides[side])
            rows.append(("JUMLAH", n, "", "", round(value, 1), "", side))

    write_csv(out / "perbandingan.csv",
              ["kategori", "n", "dasar", "konstan", "dilatih", "selisih",
               "catatan"], rows)


def table_runs(runs: dict[str, dict], out: Path) -> None:
    """The training table: how long each run took, what it cost, where loss went.

    Seconds per step rather than only total seconds, because total seconds says
    nothing until you know how many steps it bought. It is also the only column
    that stays comparable across runs of different sizes, and the one that says
    whether a slower card was slower per unit of work or merely given more work.

    Wall clock is reported next to training time. They differ by the minutes
    spent downloading and loading a checkpoint, which is real time that a reader
    reproducing this will wait through and which the billing meter counts.
    """
    rows = []
    total_s = total_usd = 0.0
    for run_id, m in sorted(runs.items()):
        lora = m.get("lora", {})
        total_s += m["train_runtime_s"]
        total_usd += m.get("estimated_cost_usd", 0) or 0
        rows.append((run_id, m["model_id"], m.get("dataset", ""),
                     m.get("tag", ""), m["examples"], m["epochs"],
                     m["steps_done"], m["seed"], m["gpu"],
                     lora.get("r", ""), lora.get("alpha", ""),
                     m.get("learning_rate", ""),
                     m["train_runtime_s"], m.get("total_s", ""),
                     m.get("seconds_per_step", ""),
                     round(m["train_runtime_s"] / 60, 1),
                     m["peak_memory_gb"], m.get("usd_per_hour", ""),
                     m["estimated_cost_usd"], m["first_loss"], m["last_loss"],
                     m.get("best_epoch", ""), m.get("train_sha256", "")[:16]))
    # A grand total, because the thesis reports the cost of the whole
    # experiment and summing a column by hand is how that number goes stale.
    if rows:
        rows.append(("TOTAL", "", "", "", "", "", "", "", "", "", "", "",
                     round(total_s, 1), "", "", round(total_s / 60, 1),
                     "", "", round(total_usd, 2), "", "", "", ""))
    write_csv(out / "latihan.csv",
              ["run_id", "model_id", "regime", "tag", "contoh", "epoch",
               "langkah", "seed", "gpu", "lora_r", "lora_alpha",
               "learning_rate", "detik_latih", "detik_total",
               "detik_per_langkah", "menit_latih", "memori_gb", "usd_per_jam",
               "usd", "loss_awal", "loss_akhir", "epoch_terbaik",
               "sha_data_latih"],
              rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("hasil"))
    ap.add_argument("--out", type=Path, default=Path("docs/gambar"))
    # One regime per run of this script, and it is not a convenience.
    #
    # hasil/ holds predictions from both test sets. They share category names
    # and a 0-100 axis, so nothing stops them being drawn as neighbouring bars
    # -- and a chart that does it is comparing scores on 766 questions with
    # scores on 1002 different questions, under a single floor line that is
    # wrong for one of them (26.9% against 33.6%). The result looks like a
    # model comparison and is an artefact of which file was globbed.
    ap.add_argument("--regime", default="split70", choices=sorted(REGIME_KEYS),
                    help="himpunan uji yang digambar; berkas regime lain diabaikan")
    args = ap.parse_args(argv)

    if not args.dir.exists():
        raise SystemExit(f"tidak ada {args.dir}/. jalankan modal_train.py::fetch dulu.")

    # Each regime writes into its own folder rather than into filenames with a
    # suffix. Same effect against collisions, but it also means a figure can
    # never be picked up for the thesis without its regime being visible in the
    # path it came from.
    args.out = args.out / args.regime
    args.out.mkdir(parents=True, exist_ok=True)

    runs = {rid: m for rid, m in manifests(args.dir).items()
            if m.get("dataset") == args.regime}
    preds = sorted(p for p in args.dir.glob(f"*--{args.regime}.jsonl"))
    if not preds:
        raise SystemExit(f"tidak ada prediksi regime {args.regime} di {args.dir}/")
    print(f"regime {args.regime}: {len(runs)} manifes, "
          f"{len(preds)} berkas prediksi\n")

    scored = {p: score_file(p) for p in preds}
    for path, tally in scored.items():
        n = sum(v[2] for v in tally.values())
        l = sum(v[1] for v in tally.values())
        print(f"  {path.name:44} {n:>5} soal  {l / n * 100:>5.1f}%"
              if n else f"  {path.name:44} kosong")
    print()

    figure_loss(runs, args.out)
    figure_categories(scored, args.out)
    figure_models(scored, args.out)
    table_decomposition(scored, args.out)
    if runs:
        table_runs(runs, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
