#!/usr/bin/env python3
"""Convert public Indonesian benchmarks into the LaguQA-MC file shape.

WHY THIS EXISTS

Fine-tuning on 10.000 examples drawn from one songbook is a narrow diet, and a
narrow diet is how a model forgets everything else. Nothing measured so far can
detect that: every number in the thesis comes from LaguQA itself, which is the
one domain the adapter was trained on, so a model that had lost all of its
chemistry and half of its Javanese would still score well and look healthy.

IndoMMLU and IndoCulture are the probes. Neither is in the training data,
neither is about songs, and both are public with published numbers, so a reader
can check the base-model column against the literature instead of taking this
repository's word for it.

WHAT THE OUTPUT IS COMPARED AGAINST

The absolute score on either benchmark is not the result. The result is the
difference between the same base model with and without the adapter, produced
by the same function on the same file. predict() with an empty run_id is the
base, so the two cannot drift apart in how they were produced -- the same
guarantee the LaguQA leaderboard already relies on.

WHY THESE FILES DO NOT LIVE IN data/benchmark/

Both sources are CC BY-NC-SA 4.0. A converted file is an adapted work and
inherits ShareAlike, which contradicts the CC BY 4.0 the LaguQA dataset release
carries. dataset_release.py picks up its payload with
BENCHMARK_DIR.glob("*.jsonl"), so a file dropped in there is published under the
wrong licence, silently, with no step in between that would notice. They go to
data/eksternal/ instead, which nothing globs.

What may be published is this converter, the frozen list of sampled ids, and
the accuracy figures. Measurements are facts about a work, not copies of it.

WHY INDOMMLU IS SUBSAMPLED AND INDOCULTURE IS NOT

IndoCulture is 2.429 items, roughly the size of the LaguQA multiple-choice set,
so it runs whole. IndoMMLU is 14.906, and every model must be run twice over it
for the two prompt conditions. A frozen stratified sample of 2.000 costs a
fifth of that and still resolves the effect being looked for: at n=2000 the
standard error of a proportion near 0.5 is about 1,1 points, while forgetting
worth reporting moves a score by ten points or more. The sample is written out
as an id list so the choice can be checked and repeated, and the full set is
still there if a result lands close enough to the noise to need it.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(10**8)

REPOS = {
    "indommlu": ("indolem/IndoMMLU", "IndoMMLU.csv"),
    "indoculture": ("indolem/IndoCulture", "test.csv"),
}

LISENSI = "CC BY-NC-SA 4.0"

# Options arrive as one string with the letters embedded, "A. Galungan\nB.
# Kuningan\n...". Anchored to line starts because option text runs on freely and
# a letter-dot pair inside a sentence would otherwise split an option in half.
OPSI_RE = re.compile(r"^([A-E])\.\s?", re.M)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subject_groups(script: Path) -> dict[str, str]:
    """Read subject2group out of the dataset's own loading script.

    Transcribing this table by hand would put twenty-five hand-typed strings
    between the source and a per-domain result, and a single typo there moves
    questions into a group they do not belong to without failing anything. The
    mapping is parsed from the file the dataset publishes, so it is the authors'
    grouping or the conversion stops.
    """
    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "subject2group" in names:
            return ast.literal_eval(node.value)
    raise SystemExit(f"{script.name} tidak memuat subject2group")


def parse_opsi(blob: str) -> dict[str, str]:
    """Split the option blob into {letter: text}, letters stripped off.

    The letter prefix has to go. Scoring is by option text: predict() appends
    each option's text to the prompt and reads its mean log-probability, while
    mc_messages() renders the letters itself. Leaving them in would show the
    model "A. A. Galungan" and score the duplicated letter as part of the
    answer.
    """
    marks = list(OPSI_RE.finditer(blob or ""))
    out: dict[str, str] = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(blob)
        teks = blob[m.end():end].strip()
        if teks:
            out[m.group(1)] = teks
    return out


def load_indommlu(path: Path, groups: dict[str, str]) -> tuple[list[dict], dict]:
    baris = list(csv.DictReader(path.open(encoding="utf-8")))
    dibuang: Counter[str] = Counter()
    out: list[dict] = []
    for r in baris:
        # The 75 few-shot rows are the prompt material the authors set aside,
        # not test questions. Keeping them would put the published 14.906 out of
        # reach and mix demonstration items into the score.
        if r["is_for_fewshot"] == "1":
            dibuang["contoh few-shot"] += 1
            continue
        opsi = parse_opsi(r["jawaban"])
        if r["kunci"] not in opsi:
            # Two rows in the release have a key pointing at an option that is
            # not there. Dropped rather than guessed, and counted so the number
            # appears in the header instead of only here.
            dibuang["kunci tidak ada di opsi"] += 1
            continue
        subject = r["subject"].strip()
        if subject not in groups:
            dibuang["subject di luar subject2group"] += 1
            continue
        # Stripped because one value ships as "VIII SMP " with a trailing
        # space. That is cosmetic, not the source of the count: subject x level
        # gives 64 combinations either way, while the paper reports 63 tasks.
        # The strata below are therefore this file's own grouping and not the
        # authors' task list, which changes nothing for a proportional sample
        # but should not be written up as if it matched.
        level = r["level"].strip()
        out.append({
            "id": f"indommlu-{int(r['id']):05d}",
            "id_lagu": f"indommlu-{int(r['id']):05d}",
            "kategori": groups[subject],
            "tingkat": level,
            "subjek": subject,
            "pertanyaan": r["soal"].strip(),
            "opsi": opsi,
            "kunci": r["kunci"],
            "sumber_kunci": "eksternal",
            "kekerasan_pengecoh": "",
        })
    return out, dict(dibuang)


def load_indoculture(path: Path) -> tuple[list[dict], dict]:
    baris = list(csv.DictReader(path.open(encoding="utf-8")))
    dibuang: Counter[str] = Counter()
    out: list[dict] = []
    for r in baris:
        pilihan = ast.literal_eval(r["options"])
        opsi = {}
        for teks in pilihan:
            teks = str(teks).strip()
            m = OPSI_RE.match(teks)
            if not m:
                continue
            opsi[m.group(1)] = teks[m.end():].strip()
        if r["answer"] not in opsi:
            dibuang["kunci tidak ada di opsi"] += 1
            continue
        out.append({
            "id": f"indoculture-{int(r['id']):05d}",
            "id_lagu": f"indoculture-{int(r['id']):05d}",
            "kategori": r["topic"].strip(),
            # The province-specific flag goes in `tingkat` because it is the
            # difficulty axis of this set: an item marked False holds anywhere
            # in Indonesia, one marked True needs the local custom.
            "tingkat": ("khas-provinsi" if r["is_province_specific"] == "True"
                        else "umum"),
            "provinsi": r["province"].strip(),
            # A premise with three continuations, not a question. The lead-in is
            # part of the item text rather than a change to mc_messages, so both
            # prompt conditions and every model see the identical user turn and
            # only the system message varies.
            "pertanyaan": f"{r['context'].strip()}\n\n"
                          f"Kalimat lanjutan yang paling wajar adalah:",
            "opsi": opsi,
            "kunci": r["answer"],
            "sumber_kunci": "eksternal",
            "kekerasan_pengecoh": "",
        })
    return out, dict(dibuang)


def stratified(items: list[dict], n: int, seed: int,
               strata) -> list[dict]:
    """Proportional sample across strata, exact total, repeatable.

    Largest-remainder allocation: every stratum gets its floor share, then the
    leftover seats go to the largest fractional parts. Proportional rather than
    equal-sized because the aggregate is meant to stand in for the full set, and
    equal sizes would reweight a benchmark whose subjects run from 106 items to
    3210.
    """
    if n >= len(items):
        return sorted(items, key=lambda x: x["id"])
    kelompok: dict[tuple, list[dict]] = defaultdict(list)
    for it in items:
        kelompok[strata(it)].append(it)
    kunci = sorted(kelompok)
    kuota = {k: n * len(kelompok[k]) / len(items) for k in kunci}
    jatah = {k: int(v) for k, v in kuota.items()}
    sisa = n - sum(jatah.values())
    # Ties broken by stratum name, so the sample does not depend on dict order.
    for k in sorted(kunci, key=lambda k: (-(kuota[k] - jatah[k]), k))[:sisa]:
        jatah[k] += 1
    rng = random.Random(seed)
    out: list[dict] = []
    for k in kunci:
        anggota = sorted(kelompok[k], key=lambda x: x["id"])
        out.extend(rng.sample(anggota, min(jatah[k], len(anggota))))
    return sorted(out, key=lambda x: x["id"])


def write_mc(path: Path, header: dict, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def ringkas(items: list[dict]) -> str:
    n_opsi = Counter(len(x["opsi"]) for x in items)
    # The random-guess floor is not 20 percent here. IndoMMLU mixes three, four
    # and five option items, so the floor is the mean of 1/k over the items
    # actually sampled, and a single hard-coded baseline would misjudge every
    # model against it.
    dasar = sum(1 / len(x["opsi"]) for x in items) / len(items)
    return (f"{len(items)} soal, opsi {dict(sorted(n_opsi.items()))}, "
            f"tebakan acak {dasar:.1%}")
