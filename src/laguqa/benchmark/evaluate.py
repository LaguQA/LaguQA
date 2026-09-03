#!/usr/bin/env python3
"""Score model answers against the LaguQA keys.

Exact string matching would mark most correct answers wrong. A model asked who
wrote Syukur replies "Lagu Syukur diciptakan oleh H. Mutahar." while the key
is "H. Mutahar"; asked for a bar count it replies "Ada 8 bar." instead of "8".
Both are right. Scoring has to look at content, not at wrapping.

Being lenient opens a different hole: if any mention of the key counts, a model
that lists every composer in the book scores full marks. So for the categories
that have a closed answer set, a prediction only counts when it contains the
key AND contains no rival answer from that same set.

Each category is scored by the rule that fits its answer shape, and every run
reports strict and lenient side by side so the gap is visible rather than
hidden behind one number.

Input is a JSONL file of predictions: one object per line with "id_lagu",
"kategori" and "prediksi", matching the items in a key file.

WHICH KEY FILE MATTERS. There are two test sets. laguqa_test.jsonl covers all
107 songs and belongs to the released model; laguqa_test_split37.jsonl covers
only the 37 held-out songs and is the one the experiment reports. They share
the (id_lagu, kategori) key space but not the questions, so scoring split37
predictions against the full keys silently pairs an answer with a question the
model was never asked. Nothing raises, and the accuracy that comes out is
meaningless. --kunci must therefore name the same file the predictions were
generated from; predict() writes that filename into the prediction file's name.

Usage:
    python scripts/11_evaluate.py predictions.jsonl
    python scripts/11_evaluate.py predictions.jsonl --kunci data/benchmark/laguqa_test_split37.jsonl
    python scripts/11_evaluate.py predictions.jsonl --per-song
    python scripts/11_evaluate.py predictions.jsonl --tulis-csv   # audit per soal
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from laguqa.paths import BENCHMARK_DIR, TEST_PATH

# Which key file each training regime's test side lives in. Mirrors DATASETS in
# modal_train.py; if one gains a regime the other has to follow.
REGIME_KEYS = {
    "split70": BENCHMARK_DIR / "laguqa_test_split37.jsonl",
    "full": TEST_PATH,
}

# A lyric or notation answer is counted correct above this token overlap. Set
# where a human would still call the answer the same line: below it the reply
# has either lost or invented a meaningful part of the text.
# Token F1 for the free-text answers, and F1 counts precision, so the framing
# an instruct model puts around an answer ("Baris ke-3 berbunyi: ...") costs it.
# That is the same shape as the truncation defect -- a rule that penalises being
# verbose rather than being wrong -- so it was measured rather than argued
# about, by rescoring the untrained baseline with a sliding window that ignores
# framing entirely.
#
# It changed almost nothing: judul_ke_baris 0/50 -> 1/50, lanjut_lirik
# 0/50 -> 1/50, judul_ke_lirik 0/37 -> 0/37. The baseline does not know these
# lyrics; it was not being robbed of them by the rule.
#
# And the one item that flipped argues for keeping plain F1. It flipped because
# the reply quoted the question's own lyric back, which a window search finds
# somewhere inside a long answer and credits as recall. Window matching would
# have made "repeat everything you were given" a scoring strategy, which is the
# failure the giveaway filter in generate.py exists to prevent. Left alone.
F1_PASS = 0.80

# Ways of declining to answer, both "the book does not say" and the plainer
# "I do not know". Both were widened to include the second kind after the
# baseline model answered "Mohon maaf, saya tidak memiliki informasi spesifik
# mengenai pencipta lagu X" and scored zero on every abstain item. That reply
# is the behaviour these items exist to reward: it declines to invent a
# composer. Reading it as a failure would have understated the untrained model
# on exactly the axis the abstain items were built to measure.
REFUSAL = ("tidak mencantumkan", "tidak tercantum", "tidak disebut",
           "tidak disebutkan", "tidak diketahui", "tidak ada keterangan",
           "tidak memiliki informasi", "tidak punya informasi",
           "tidak menemukan", "tidak dapat memastikan", "tidak bisa memastikan",
           "belum diketahui", "tidak ditemukan", "anonim", "maaf")


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_reasoning(pred: str) -> str:
    """Remove a reasoning model's <think> block before anything else looks at it.

    Some of the models measured here open every answer with their working.
    Scoring that working would break the comparison in both directions. The
    rival rule below marks a reply wrong for naming a competing answer, and
    thinking out loud is exactly that: a model that reasons "could be Bali or
    Timor, but the book says Timor" would be marked wrong for having weighed
    Bali. In the other direction, a model that mentions the right answer only
    to reject it would be marked right.

    An unterminated block means generation stopped mid-thought and no answer
    was ever reached, so nothing is left to score and the reply is wrong. That
    is the honest outcome, not a hole to patch.
    """
    out = THINK_RE.sub(" ", pred)
    return "" if "<think>" in out.lower() else out.strip()


def reasoning_of(pred: str) -> str:
    """The reasoning a model showed, kept so the audit file can display it.

    The scorer must not read this, but a reader checking a disputed item has
    to be able to see why the model answered as it did. Keeping it in its own
    column means the trace is preserved without ever reaching the scoring
    path. An unterminated block is returned as-is, since a run that stopped
    mid-thought is exactly the case someone will want to inspect.
    """
    blok = [m.group(1) for m in
            re.finditer(r"<think>(.*?)</think>", pred, re.DOTALL | re.IGNORECASE)]
    if not blok and "<think>" in pred.lower():
        potong = re.split(r"<think>", pred, maxsplit=1, flags=re.IGNORECASE)
        return potong[1].strip() if len(potong) > 1 else ""
    return "\n---\n".join(b.strip() for b in blok)


def normalise(s: str) -> str:
    """Fold case, accents and punctuation so only the wording is compared."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("-", " ").replace("'", " ")
    s = re.sub(r"[^\w\s/#,]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokens(s: str) -> list[str]:
    return normalise(s).split()


def f1(pred: str, gold: str) -> float:
    """Token overlap, the usual measure for free-form text answers."""
    p, g = tokens(pred), tokens(gold)
    if not p or not g:
        return float(p == g)
    common = 0
    pool = list(p)
    for t in g:
        if t in pool:
            pool.remove(t)
            common += 1
    if not common:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


# Where a letter in a reply is actually naming a key, rather than being a
# letter. Matched against normalised text, where "Do = C" has become "do c".
#
# The whole of key_signature in this dataset is six bare letters -- C, F, E, G,
# D, A, no sharps or flats -- so nothing more elaborate is needed, and anything
# more elaborate would be guessing at data that does not exist.
KEY_CONTEXT = re.compile(
    r"(?:do|nada dasar(?:nya)?|bernada dasar|kunci|key)\s+([a-g])(?![a-z])"
    r"|(?<![a-z])([a-g])\s+(?:mayor|major|minor|mol)(?![a-z])"
)


def keys_named(pred: str) -> set[str]:
    """Every key the reply actually names, however it spells it.

    The gold answers are written in the book's jianpu convention, "Do = C". A
    model that replies "C Mayor" has named the same key and was being scored
    wrong for not using the convention -- and only the fine-tuned model ever
    learns the convention, because it is in the training data. That turns 50 of
    the 1002 questions into a test of formatting that the baseline cannot pass
    and the trained model cannot fail, which widens the gap this thesis is
    trying to measure honestly.

    Returning a set rather than a first match keeps the hedge rule working:
    "C Mayor atau G Mayor" names two keys, so it has not answered.
    """
    flat = normalise(pred)
    found = {a or b for a, b in KEY_CONTEXT.findall(flat)}
    if not found and re.fullmatch(r"[a-g]", flat):
        found = {flat}
    return found


def whole_word(needle: str, haystack: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


# Titles the source book spells more than one way. Song 76 is headed
# "BHINEKA TUNGGAL IKA" on both its own page and in the table of contents, but
# the lyric printed underneath the melody reads "Bhin-ne-ka". The book
# contradicts itself, so a model is right either way and neither spelling may
# be counted as an error. Keys are already normalised.
SPELLINGS: dict[str, tuple[str, ...]] = {
    "bhinneka tunggal ika": ("bhineka tunggal ika",),
}


def variants(gold: str) -> tuple[str, ...]:
    return (gold,) + SPELLINGS.get(gold, ())


def contains_key(pred: str, gold: str, rivals: set[str]) -> bool:
    """Key present and no competing answer present.

    The second half is what stops a model from scoring by listing every
    candidate it can think of.

    A rival that is itself part of the key is skipped. Otherwise answering
    "Desaku Yang Kucinta" would be marked wrong because the shorter title
    "Desaku" is also in the pool and is unavoidably present inside it.
    """
    p, g = normalise(pred), normalise(gold)
    if not any(whole_word(v, p) for v in variants(g)):
        return False
    for other in rivals:
        if not other or other == g or other in g:
            continue
        if whole_word(other, p):
            return False
    return True


VERDICT_RE = re.compile(r"(?:(tidak|bukan|kurang|belum)\s+)?(benar|tepat|salah|keliru)")


def verdict(s: str) -> str | None:
    """The first true/false judgement in a reply, with negation applied.

    Needed because "tidak benar" contains "benar": matching on the bare word
    would read a rejection as an agreement.
    """
    m = VERDICT_RE.search(normalise(s))
    if not m:
        return None
    positive = m.group(2) in ("benar", "tepat")
    if m.group(1):
        positive = not positive
    return "benar" if positive else "salah"


def numbers(s: str) -> list[str]:
    return re.findall(r"-?\d+", s.replace(",", ""))


def number_matches(pred: str, gold: str) -> bool:
    """Accept the first or the last number in the reply.

    A model that works aloud puts the answer last ("bar 1, bar 2, jadi 2 bar");
    one that answers flatly puts it first ("2 bar"). Both are the same answer.
    """
    p, g = numbers(pred), numbers(gold)
    if not p or not g:
        return False
    return g[0] in (p[0], p[-1])


def jianpu_tokens(s: str) -> list[str]:
    """Keep only what carries the melody, so spacing and prose do not count."""
    return re.findall(r"[#b]*[0-7][',]*_*\.?|-|\|", s)


# --- per-category scorers ----------------------------------------------------


def score(kategori: str, pred: str, gold: str, rivals: set[str]) -> tuple[bool, bool]:
    """Return (strict, lenient)."""
    pred = strip_reasoning(pred)
    strict = normalise(pred) == normalise(gold)

    if kategori.endswith("abstain"):
        # Correct means admitting the book does not say AND not naming anyone
        # anyway. The second half used to be a promise in this comment that the
        # code did not keep, so "Buku tidak mencantumkan penciptanya, tetapi
        # kemungkinan Ismail Marzuki" scored full marks -- which is precisely
        # the hedge-then-invent behaviour these items exist to catch.
        said_no = any(k in pred.lower() for k in REFUSAL)
        if not said_no:
            return strict, False
        # A rival here is a real answer from the positive twin of this category
        # (pencipta for pencipta_abstain), so naming one is naming a composer.
        named = any(whole_word(other, normalise(pred))
                    for other in rivals if other and len(other) > 3)
        return strict, not named

    if kategori.startswith("verifikasi"):
        # Only the verdict is scored; the explanation after it varies freely.
        return strict, verdict(pred) is not None and verdict(pred) == verdict(gold)

    if kategori in {"hitung_bar", "ambitus"}:
        return strict, number_matches(pred, gold)

    if kategori == "nada_dasar":
        want, got = keys_named(gold), keys_named(pred)
        return strict, bool(want) and got == want

    if kategori in {"judul_ke_notasi", "nada_tertinggi", "nada_terendah"}:
        p, g = jianpu_tokens(pred), jianpu_tokens(gold)
        if kategori != "judul_ke_notasi":
            return strict, bool(p) and p[0] == (g[0] if g else None)
        if not g:
            return strict, False
        return strict, SequenceMatcher(None, p, g).ratio() >= F1_PASS

    if kategori in {"judul_ke_lirik", "judul_ke_baris", "lanjut_lirik"}:
        return strict, f1(pred, gold) >= F1_PASS

    # Short closed-set answers: title, composer, region, key, tempo, meter,
    # type, and the single word of a cloze.
    return strict, contains_key(pred, gold, rivals)


# --- driver ------------------------------------------------------------------


def load_keys(path: Path = TEST_PATH) -> dict[tuple[str, str], list[dict]]:
    items = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for x in items:
        out[(x["id_lagu"], x["kategori"])].append(x)
    return out


def rows_of(path: Path) -> list[dict]:
    """Prediction rows, skipping the version header if one is present."""
    return [d for d in (json.loads(l) for l in
                        path.read_text(encoding="utf-8").splitlines() if l.strip())
            if "id_lagu" in d]


def check_version(pred: Path, keys: Path) -> None:
    """Refuse answers written against a different build of the question file.

    Slots survive a rebuild -- same songs, same categories, same count -- so
    without this the scorer pairs slot to slot and reports an ordinary-looking
    number for answers given to different questions. A stale file was caught
    once only because its sampled songs happened to differ, which is luck.
    Files written before the header existed carry no hash and are let through;
    those belong in hasil/arsip/.
    """
    baris = pred.read_text(encoding="utf-8").splitlines()
    if not baris:
        return
    d = json.loads(baris[0])
    if "sha256" not in d or "id_lagu" in d:
        return
    ada = hashlib.sha256(keys.read_bytes()).hexdigest()
    if d["sha256"] != ada:
        raise SystemExit(
            f"{pred.name} menjawab {keys.name} versi lain.\n"
            f"  prediksi dibuat atas : {d['sha256'][:16]}\n"
            f"  berkas soal sekarang : {ada[:16]}\n"
            f"Jalankan ulang predict(), atau pindahkan ke hasil/arsip/.")


def key_order(path: Path) -> list[tuple[str, str]]:
    return [(x["id_lagu"], x["kategori"]) for x in rows_of(path)]


def check_alignment(pred_slots: list[tuple[str, str]],
                    key_slots: list[tuple[str, str]]) -> str:
    """Predictions must arrive in the order the test file asked the questions.

    Where a song and category hold more than one question, the loop below tells
    them apart by position and nothing else -- there is no question id in a
    prediction file. So a file that is complete, correctly formed, and merely
    shuffled scores every one of those items against a sibling's key and
    reports a plausible number.

    predict() promises to write rows in test-file order. That is a comment;
    this is the check of it.

    WHAT THIS CANNOT SEE

    A prediction carries no question id, so two questions swapped INSIDE one
    slot are invisible here -- both files still read (song, category) in the
    same sequence. That permutation needs a sort keyed on the slot itself,
    which nothing does; every plausible reordering (by category, by song, by
    length) moves items across slots and is caught. Worth knowing rather than
    trusting: closing the gap properly means putting a question id in the
    prediction schema, and that schema is frozen and hashed.
    """
    if pred_slots == key_slots:
        return ""
    if sorted(pred_slots) == sorted(key_slots):
        return ("PERINGATAN: prediksi lengkap tetapi URUTANNYA berbeda dari "
                "berkas kunci. Soal yang satu lagu-kategorinya lebih dari satu "
                "akan dinilai dengan kunci saudaranya. Skor di bawah tidak sah.")
    return ""


def rival_sets(keys: dict) -> dict[str, set[str]]:
    """All answers seen for a category, used as the distractor pool.

    An abstain category is given its POSITIVE twin's answers instead of its
    own. Its own answers are all sentences saying the book is silent, which
    are useless as distractors; what the abstain scorer needs to detect is a
    reply that declines and then names a composer anyway, and the names live
    in "pencipta", not in "pencipta_abstain".
    """
    out: dict[str, set[str]] = defaultdict(set)
    for (_, kategori), items in keys.items():
        for x in items:
            out[kategori].add(normalise(x["messages"][2]["content"]))

    for kategori in [k for k in out if k.endswith("_abstain")]:
        twin = kategori[: -len("_abstain")]
        out[kategori] = set(out.get(twin, set()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", help="JSONL: id_lagu, kategori, prediksi")
    ap.add_argument("--kunci", type=Path, default=None,
                    help="berkas kunci; bawaannya ditebak dari nama berkas prediksi")
    ap.add_argument("--per-song", action="store_true")
    ap.add_argument("--tulis-md", action="store_true",
                    help="tulis tabel skor untuk kartu model")
    ap.add_argument("--tulis-csv", action="store_true",
                    help="tulis satu baris per soal: pertanyaan, kunci, "
                         "jawaban model, dan putusannya")
    args = ap.parse_args()

    # predict() names its output "<run>--<regime>.jsonl", so the regime says
    # which of the two test sets the questions came from. Guessing it here
    # rather than defaulting to laguqa_test.jsonl means the common case is right
    # without anyone having to remember the flag.
    key_path = args.kunci
    if key_path is None:
        regime = Path(args.predictions).stem.rsplit("--", 1)[-1]
        guess = REGIME_KEYS.get(regime)
        key_path = guess if guess else TEST_PATH
    print(f"kunci: {key_path.name}")

    keys = load_keys(key_path)
    rivals = rival_sets(keys)
    used: dict[tuple[str, str], int] = defaultdict(int)
    missing = 0

    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # strict, lenient, n
    per_song: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    audit: list[dict] = []

    check_version(Path(args.predictions), key_path)
    preds = rows_of(Path(args.predictions))

    complaint = check_alignment([(p["id_lagu"], p["kategori"]) for p in preds],
                                key_order(key_path))
    if complaint:
        print(f"\n{complaint}\n")

    for p in preds:
        slot = (p["id_lagu"], p["kategori"])
        bucket = keys.get(slot)
        if not bucket:
            missing += 1
            if missing <= 5:
                print(f"[LEWATI] tidak ada kunci untuk lagu {p['id_lagu']} / {p['kategori']}")
            continue
        # Several items can share a song and category; they are consumed in
        # file order so each prediction is matched to its own key.
        gold_item = bucket[min(used[slot], len(bucket) - 1)]
        used[slot] += 1
        gold = gold_item["messages"][2]["content"]

        s, l = score(p["kategori"], p.get("prediksi", ""), gold, rivals[p["kategori"]])
        audit.append({
            "id_lagu": p["id_lagu"],
            "kategori": p["kategori"],
            "sumber_kunci": gold_item.get("sumber_kunci", ""),
            "pertanyaan": gold_item["messages"][1]["content"],
            "kunci": gold,
            "prediksi": p.get("prediksi", ""),
            "tepat": int(s),
            "toleran": int(l),
        })
        t = tally[p["kategori"]]
        t[0] += s
        t[1] += l
        t[2] += 1
        per_song[p["id_lagu"]][0] += l
        per_song[p["id_lagu"]][1] += 1

    # Refused before anything is printed, because the warning used to come
    # after a complete-looking table and a JUMLAH line, and a number that has
    # already been read is a number that can be copied into the thesis. A
    # prediction file for this benchmark has a key for every row -- a --limit
    # run answers a prefix, so it is short, never unmatched. Any unmatched row
    # means the file answers a different build, which is how a headerless v1.3
    # file restored by fetch() got as far as printing 18.1 percent.
    if missing:
        share = missing / (missing + sum(t[2] for t in tally.values())) * 100
        raise SystemExit(
            f"\n{missing} dari {missing + sum(t[2] for t in tally.values())} "
            f"prediksi ({share:.1f}%) tidak punya kunci di {key_path.name}.\n"
            f"Berkas ini menjawab bangunan soal yang lain, jadi tidak dinilai.\n"
            f"Jalankan ulang predict(), atau pindahkan ke hasil/arsip/.")

    print(f"{'kategori':24} {'n':>5} {'tepat':>8} {'toleran':>9}")
    tot = [0, 0, 0]
    for k in sorted(tally):
        s, l, n = tally[k]
        tot = [tot[0] + s, tot[1] + l, tot[2] + n]
        print(f"{k:24} {n:>5} {s / n * 100:>7.1f}% {l / n * 100:>8.1f}%")
    if tot[2]:
        print(f"{'JUMLAH':24} {tot[2]:>5} {tot[0] / tot[2] * 100:>7.1f}% "
              f"{tot[1] / tot[2] * 100:>8.1f}%")

    if args.per_song:
        print("\nper lagu (toleran):")
        for sid in sorted(per_song, key=int):
            benar, n = per_song[sid]
            print(f"  {sid:>3}  {benar:>3}/{n:<3} {benar / n * 100:>5.1f}%")

    if args.tulis_md and tot[2]:
        write_markdown(Path(args.predictions), key_path, tally, tot)
    if args.tulis_csv and audit:
        summarise_errors(audit)
        print(f"\nditulis {write_audit(Path(args.predictions), audit)}")
    return 0


def write_audit(predictions: Path, rows: list[dict]) -> Path:
    """One row per question: what was asked, the key, the answer, the verdict.

    The aggregate tables say a category scored 12 percent. They cannot say
    whether the model was wrong, right in a wording the scorer rejected, or
    answering a question whose key is itself wrong -- and those three call for
    three different responses. Two defects in this project were found exactly
    that way: the tempo key that was a bare dash, and the lyric questions that
    contained their own answers.

    Written as CSV so it opens in a spreadsheet, since the person auditing this
    is reading it as evidence for a thesis, not piping it anywhere. Newlines
    inside lyrics and notation are real content, so the CSV writer quotes them
    and the file must be read with a CSV reader, not line by line.
    """
    out = predictions.with_suffix(".audit.csv")
    fields = ["id_lagu", "kategori", "sumber_kunci", "pertanyaan", "kunci",
              "prediksi", "tepat", "toleran"]
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return out


def summarise_errors(rows: list[dict]) -> None:
    """Where the misses are, and what the model said instead.

    Printed rather than only written, because a category at 0 percent is the
    signal worth acting on and it should not need a spreadsheet to be seen. A
    single repeated wrong answer means the model settled on one guess; many
    different wrong answers mean it is trying and failing, and those are not
    the same problem.
    """
    per: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if not r["toleran"]:
            per[r["kategori"]].append(r)
    if not per:
        return
    print("\nkesalahan tersering per kategori")
    for kategori in sorted(per, key=lambda k: -len(per[k]))[:8]:
        salah = per[kategori]
        umum = Counter(r["prediksi"].strip()[:60] for r in salah).most_common(1)[0]
        unik = len({r["prediksi"].strip() for r in salah})
        print(f"  {kategori:22} {len(salah):4} salah, {unik:4} jawaban berbeda")
        print(f"      paling sering ({umum[1]}x): {umum[0]!r}")


def write_markdown(predictions: Path, key_path: Path,
                   tally: dict[str, list[int]], tot: list[int]) -> None:
    """Write the score table beside the predictions, for the model card.

    release() looks for exactly this filename and pastes it into the card's
    Hasil section. Generated rather than typed so a published card cannot
    advertise a number the scorer never produced.

    The regime stays in the name. Dropping it, as this did at first, means
    "gemma4-e2b-base" scored on split37 and the same checkpoint scored on the
    full set write to one file, and the second silently replaces the first --
    two different measurements on two different question sets, with nothing in
    the surviving file to say which one it is.
    """
    out = predictions.with_name(predictions.stem + "--skor.md")
    lines = [
        f"Dinilai atas `{key_path.name}` ({tot[2]} soal) dengan "
        f"`scripts/11_evaluate.py`.",
        "",
        "| kategori | n | tepat | toleran |",
        "|---|---|---|---|",
    ]
    for k in sorted(tally):
        s, l, n = tally[k]
        lines.append(f"| {k} | {n} | {s / n * 100:.1f}% | {l / n * 100:.1f}% |")
    lines += [
        f"| **JUMLAH** | **{tot[2]}** | **{tot[0] / tot[2] * 100:.1f}%** | "
        f"**{tot[1] / tot[2] * 100:.1f}%** |",
        "",
        "Bandingkan dengan lantai kontrol, bukan dengan nol: penebak yang "
        "selalu menjawab nilai tersering mendapat 27,5% toleran tanpa "
        "mengenal satu lagu pun. Lihat `scripts/17_controls.py`.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nditulis {out}")


if __name__ == "__main__":
    sys.exit(main())
