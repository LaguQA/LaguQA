#!/usr/bin/env python3
"""Build the LaguQA conversational training and test sets.

Reads the frozen split and turns 107 songs into chat examples. The two sides
are built separately and from separate template pools: if the test questions
were worded exactly like the training ones, the score would only measure
template matching.

Two things decide whether this works at all.

First, where the fact sits. Loss is computed on the assistant turn only, so a
fact the model should memorise has to be the ANSWER, not the context. Putting
the ABC in the user turn teaches the model to read notation; putting it in the
assistant turn teaches it to recall the song. Memorisation items therefore
answer with the notation, reasoning items are asked about it.

Second, abstention. 52 songs have no composer printed and 52 have no region.
If every training answer is a positive fact, the model learns that questions
about Indonesian songs always have an answer, and invents composers for
anonymous folk songs. The honest "the source book does not name a composer"
items are the antidote.

Question text is Indonesian because that is the language of the benchmark.

Usage:
    python scripts/10_generate_benchmark.py
    python scripts/10_generate_benchmark.py --apply
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from laguqa.notation.abc_to_jianpu import convert as to_jianpu
from laguqa.paths import (
    CSV_PATH,
    SPLIT_PATH,
    TEST_PATH as OUT_TEST,
    TRAIN_PATH as OUT_TRAIN,
)

csv.field_size_limit(10**8)

SYSTEM = (
    "Kamu asisten yang menguasai lagu nasional dan lagu daerah Indonesia, "
    "termasuk notasi angka dan notasi ABC-nya."
)

# Two regimes, because the two questions need different splits.
#
# "all" trains on every song. That is the deliverable: knowledge injection,
# where withholding a song would mean the model cannot possibly answer about
# it. What is held out instead is the FORM — the wording, the excerpt, the
# lyric line, the blank position. The fact is taught one way and asked
# another, which is what a retention test should do.
#
# "split" keeps the frozen 70/37 song split. It is the only honest way to ask
# whether the model learned to READ notasi angka rather than memorise it,
# because the 37 songs were never seen in any form.
# 0 means "keep the whole pool", which is the recipe both regimes now use.
# See sample_to_target for why trimming was removed.
TARGET_ALL = 0
TARGET_SPLIT = 0
TEST_PER_CATEGORY = 50
SEED = 20260901

EXCERPT_BARS = (2, 4, 8)

# Lengths for hitung_bar only. Wider and denser than EXCERPT_BARS so the answer
# is not one of three memorised values; see reasoning_items.
COUNT_BARS = tuple(range(3, 13))

# (category, question, answer, key source, unit)
#
# The unit names the piece of the song an item consumes, so the test side can
# claim it and the training side can then avoid it. None means the item has no
# withholdable content: a song has one composer, and the only way to separate
# those two sides is by wording.
Item = tuple[str, str, str, str, "tuple | None"]


# --- helpers -----------------------------------------------------------------


def jianpu_bars(abc: str) -> list[str]:
    """Split a tune's jianpu rendering into bars, dropping empty ones."""
    text = " ".join(to_jianpu(abc))
    # "|]" closes the tune; splitting on the bar leaves a stray "]" that is
    # not part of any bar's content.
    bars = [b.strip(" ]").strip() for b in re.split(r"\|+:?|:\|", text)]
    return [b for b in bars if b and not b.isspace()]


def lyric_lines(song: dict) -> list[str]:
    return [ln.strip() for ln in song["lyrics"].splitlines() if ln.strip()]


JIANPU_NOTE = re.compile(r"(?P<acc>[#b]*)(?P<deg>[1-7])(?P<oct>['\,]*)")
MAJOR_SEMITONES = [0, 2, 4, 5, 7, 9, 11]


def jianpu_pitches(excerpt: str) -> list[tuple[int, str]]:
    """Read an excerpt back as (relative semitone, written token).

    The questions show notasi angka, so the answers are computed from the same
    notation the reader sees rather than from the ABC behind it. That keeps a
    question and its key impossible to disagree.
    """
    out = []
    for tok in excerpt.split():
        if tok.startswith("0") or tok in {"-", "|"} or tok.startswith("("):
            continue
        m = JIANPU_NOTE.match(tok)
        if not m:
            continue
        deg = int(m.group("deg"))
        alter = m.group("acc").count("#") - m.group("acc").count("b")
        octave = m.group("oct").count("'") - m.group("oct").count(",")
        value = octave * 12 + MAJOR_SEMITONES[deg - 1] + alter
        out.append((value, m.group("acc") + m.group("deg") + m.group("oct")))
    return out


# --- memorisation items ------------------------------------------------------


def naming(song: dict) -> str:
    """The name to use when a QUESTION has to say which song it means.

    Two songs are printed as "Desaku", so "Lagu Desaku diciptakan oleh siapa?"
    was generated twice with opposing keys -- one naming L. Manik, one saying
    the book gives no composer. No model can be right on both.

    This returns title_unique, which appends the section the book files the
    song under. It is used only to ask. Where the title is the ANSWER, the
    printed title is used instead: a model that replies "Desaku" is right, and
    should not be asked to reproduce a parenthetical this project invented.

    Falls back to title so an older laguqa.csv still generates.
    """
    return (song.get("title_unique") or "").strip() or song["title"]


# A cell can be blank in more than one way. The transcription used "-" for a
# field the page leaves out, which is not blank to Python but is blank to a
# reader, and treating the two differently is what let a tempo question be
# generated with "-" as its answer.
MISSING = {"", "-", "–", "—"}


def has(song: dict, column: str) -> bool:
    return song[column].strip() not in MISSING


def metadata_items(song: dict, pool: dict, extra: dict | None = None) -> list[Item]:
    """(category, question, answer, key source, unit) for the printed metadata.

    Songs without a composer or region get an honest refusal instead of being
    skipped. Those are the items that stop the model inventing an author for
    an anonymous folk song.

    The unit is None throughout: a song has exactly one composer, and holding
    it out of training would mean never teaching it. For these the two sides
    are separated by wording alone, which is the intended test — the fact was
    taught one way and is asked another.

    On the training side `extra` supplies further wordings, so a fact that
    exists once in the book still arrives at the model several times. See
    PARAPHRASE_TRAIN for why that is not padding.
    """
    t = naming(song)
    out: list[Item] = []

    def emit(kategori: str, template: str, answer: str) -> None:
        for tanya in askings(pool, extra, template):
            out.append((kategori, tanya.format(judul=t), answer, "tercetak", None))

    if has(song, "composer"):
        emit("pencipta", "pencipta", song["composer"])
    else:
        emit("pencipta_abstain", "pencipta",
             f"Buku sumber tidak mencantumkan pencipta lagu {t}. "
             f"Lagu ini tercatat sebagai lagu daerah"
             + (f" dari {song['origin']}." if has(song, "origin") else "."))

    if has(song, "origin"):
        emit("asal", "asal", song["origin"])
    else:
        emit("asal_abstain", "asal",
             f"Buku sumber tidak mencantumkan daerah asal lagu {t}; "
             f"lagu ini tergolong lagu nasional.")

    emit("nada_dasar", "nada_dasar", f"Do = {song['key_signature']}")

    # Two songs carry "-" in the tempo column, meaning the page prints no tempo
    # marking at all. Asking about them anyway produced an item whose gold
    # answer was the single character "-", which normalises to the empty
    # string: a model that said nothing scored it correct. The negative control
    # that answers "" everywhere is what surfaced this, and the same shape of
    # bug is why composer and region already have abstain variants.
    if not has(song, "tempo"):
        emit("tempo_abstain", "tempo",
             f"Buku sumber tidak mencantumkan keterangan tempo untuk lagu {t}.")
    else:
        emit("tempo", "tempo", song["tempo"])
    emit("jenis", "jenis", song["song_type"])

    # Meter is only asked where the page actually prints it. For the other 50
    # songs the value in M: was inferred by a model, so using it as a gold
    # answer would test a model against another model's guess.
    if song["time_signature_source"] == "tercetak":
        emit("birama", "birama", song["time_signature"])
    return out


def notation_items(song: dict, pool: dict, rng: random.Random,
                   limit: int) -> list[Item]:
    """Excerpt of notasi angka -> title. Only from verified transcriptions.

    The unit is the excerpt itself, so a phrase asked at test time was never
    mapped to its title during training. The melody is still taught, through
    judul_ke_notasi, which is what makes the test answerable at all.
    """
    if song["abc_status"] != "terverifikasi":
        return []
    bars = jianpu_bars(song["abc_notation"])
    out: list[Item] = []
    for n in EXCERPT_BARS:
        if len(bars) < n:
            continue
        starts = list(range(0, len(bars) - n + 1))
        rng.shuffle(starts)
        for s in starts[:limit]:
            excerpt = " | ".join(bars[s:s + n])
            notes = jianpu_pitches(excerpt)
            # A phrase of two held notes identifies nothing. Asking which song
            # it comes from would have no single right answer, so the excerpt
            # has to carry enough shape to be worth asking about.
            if len(notes) < 4 or len({p for p, _ in notes}) < 3:
                continue
            out.append(("notasi_ke_judul",
                        pool["notasi_ke_judul"].format(notasi=excerpt),
                        song["title"], "dihitung", ("excerpt", excerpt)))
    return out


def recall_items(song: dict, pool: dict, limit: int,
                 extra: dict | None = None) -> list[Item]:
    """Title -> notation, and title -> lyric. The strongest memorisation shape.

    Recognising an excerpt only needs the model to match; producing the
    notation on request is what "knowing the song" actually means, and it is
    the direction the benchmark asks about. The answer is the notation, so the
    loss lands on it.

    judul_ke_lirik carries the whole lyric and judul_ke_notasi carries the
    whole melody. These are the items that put the song into the model, so
    judul_ke_lirik is never withheld. Individual bar ranges can be, because
    the same bars reappear inside the wider ranges of the other excerpt sizes.
    """
    out: list[Item] = []
    lines = lyric_lines(song)
    if lines:
        # One per song, like the metadata facts, and paraphrased for the same
        # reason: the whole lyric is the single richest thing the model can be
        # taught about a song, and it was being shown once.
        for tanya in askings(pool, extra, "judul_ke_lirik"):
            out.append(("judul_ke_lirik", tanya.format(judul=naming(song)),
                        song["lyrics"].strip(), "tercetak", None))
        for i, ln in enumerate(lines[:limit], start=1):
            out.append(("judul_ke_baris",
                        pool["judul_ke_baris"].format(judul=naming(song), n=i),
                        ln, "tercetak", ("line", song["id"], i - 1)))

    if song["abc_status"] != "terverifikasi":
        return out
    bars = jianpu_bars(song["abc_notation"])
    for n in EXCERPT_BARS:
        for s in range(0, len(bars) - n + 1, n):
            out.append(("judul_ke_notasi",
                        pool["judul_ke_notasi"].format(judul=naming(song),
                                                       a=s + 1, b=s + n),
                        " | ".join(bars[s:s + n]), "dihitung",
                        ("bars", song["id"], s, n)))
    return out


def drop_ambiguous(items: list[dict], all_songs: list[dict]) -> list[dict]:
    """Remove excerpt-identification items whose excerpt fits several songs.

    Checked against all 107 songs, not just this side: an excerpt shared with
    a song on the other side has no single right answer either, even though
    the other title never appears in this file.
    """
    owners: dict[str, set[str]] = {}
    for song in all_songs:
        if song["abc_status"] != "terverifikasi":
            continue
        bars = jianpu_bars(song["abc_notation"])
        for n in EXCERPT_BARS:
            for s in range(0, len(bars) - n + 1):
                owners.setdefault(" | ".join(bars[s:s + n]), set()).add(song["title"])

    out = []
    for x in items:
        if x["kategori"] == "notasi_ke_judul":
            excerpt = x["messages"][1]["content"].split("\n", 1)[-1]
            if len(owners.get(excerpt, {""})) > 1:
                continue
        out.append(x)
    return out


# 1.5 rebuilt on the composer-normalised CSV. Three songs whose composer the
# book records as "NN" now have an empty composer, so they moved from the
# pencipta questions to the abstain ones, and the several spellings of Ismail
# Marzuki became one. See laguqa.dataset.composers.
VERSION = "1.5"
MANIFEST_PATH = OUT_TRAIN.parent / "laguqa_manifest.json"


def write_manifest(regime: str, paths: list[Path], counts: dict) -> str:
    """Record what was written, so a later change to these files is detectable.

    Until now only the multiple-choice set carried a hash. The four
    conversational files had none, which meant they could drift -- through a
    regenerate, an edit, a half-finished experiment -- and nothing would say so.
    The thesis quotes numbers computed from these files; a number whose input
    cannot be identified is not a result.

    Merged rather than overwritten, because the two regimes are generated by
    separate runs and each knows only about its own half.
    """
    entry = {
        "versi": VERSION,
        "benih": SEED,
        "dihasilkan_dari": CSV_PATH.name,
        "jumlah": counts,
        "sha256": {},
    }
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entry["sha256"][path.name] = digest

    existing = {}
    if MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    existing[regime] = entry
    MANIFEST_PATH.write_text(json.dumps(existing, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    return json.dumps(entry["sha256"], ensure_ascii=False)


def drop_clashing(items: list[dict]) -> tuple[list[dict], list[str]]:
    """Remove questions that appear more than once with different answers.

    A repeated chorus makes "continue the lyric after this line" ambiguous: the
    same line has two different successors and both are right. On the test side
    that would guarantee a lost point whatever the model says; on the training
    side it is contradictory supervision. Neither is worth keeping over a
    handful of items.

    The report is the useful part. This is what found the transcription error
    in song 66, where the same line of Syukur ended three different ways and
    one of them, "semjukkan", is not a word. A clash is sometimes a duplicated
    lyric and sometimes a typo in the dataset, and only looking tells you which.
    """
    answers: dict[str, set[str]] = {}
    for x in items:
        answers.setdefault(x["messages"][1]["content"], set()).add(
            x["messages"][2]["content"])
    clashing = {q for q, a in answers.items() if len(a) > 1}
    notes = [f"{x['kategori']} lagu {x['id_lagu']}: "
             f"{x['messages'][1]['content'].splitlines()[-1][:60]!r}"
             for x in items if x["messages"][1]["content"] in clashing]
    kept = [x for x in items if x["messages"][1]["content"] not in clashing]
    return kept, sorted(set(notes))


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def gives_away(answer: str, prompt_text: str) -> bool:
    """Does the text shown in the question already contain the answer?

    Compared the way the scorer compares -- case-folded, accent-stripped,
    punctuation removed, on whole-word boundaries -- because a question is only
    self-answering if the SCORER would accept the giveaway, not if a human
    would notice the resemblance.
    """
    def flat(s: str) -> str:
        s = strip_accents(s).lower()
        return " " + re.sub(r"[^a-z0-9]+", " ", s).strip() + " "

    return flat(answer).strip() != "" and flat(answer) in flat(prompt_text)


# Function words. Blanking one tests Indonesian grammar, not knowledge of the
# song: "Ketuhanan ____ Maha Esa" is answerable as "yang" by anyone who speaks
# the language and has never heard the song. Twenty percent of cloze items had
# an answer that also appeared elsewhere in the very line being shown, and they
# were nearly all words from this list.
STOPWORDS = {
    "yang", "di", "ke", "dari", "dan", "atau", "itu", "ini", "pada", "untuk",
    "dengan", "akan", "ada", "aku", "kau", "kita", "kami", "mu", "ku", "nya",
    "tak", "tidak", "sudah", "telah", "juga", "saja", "pun", "lah", "kah",
    "para", "se", "si", "oh", "o", "hai", "la", "sang", "adalah", "dalam",
}


def is_content_word(word: str) -> bool:
    bare = re.sub(r"[^\w'-]", "", word).lower()
    return len(bare) > 2 and bare not in STOPWORDS


def lyric_items(song: dict, pool: dict, rng: random.Random,
                limit: int) -> list[Item]:
    lines = lyric_lines(song)
    out: list[Item] = []
    sid = song["id"]

    # A line that contains the title cannot be asked about. Indonesian folk
    # titles are usually a phrase lifted straight out of the song, so "which
    # song has the line 'Mana di mana anak kambing saya'" answers itself for
    # "Anak Kambing Saya". Seventeen of fifty such items were self-answering,
    # and the untrained model scored exactly 17/50 = 34.0 percent on this
    # category by quoting the prompt back. That looked like world knowledge and
    # was arithmetic.
    for i, ln in enumerate(lines[:limit]):
        if gives_away(song["title"], ln):
            continue
        out.append(("lirik_ke_judul", pool["lirik_ke_judul"].format(lirik=ln),
                    song["title"], "tercetak", ("line", sid, i)))

    for i in range(min(len(lines) - 1, limit)):
        out.append(("lanjut_lirik",
                    pool["lanjut_lirik"].format(judul=naming(song), lirik=lines[i]),
                    lines[i + 1], "tercetak", ("pair", sid, i)))

    # Several blank positions per line rather than one. The same line with a
    # different word removed is a genuinely different item: it forces recall
    # of a different part of the lyric, not a reworded version of the same one.
    # The blank position is part of the unit for the same reason.
    for i, ln in enumerate(lines[:limit]):
        words = ln.split()
        if len(words) < 4:
            continue
        spots = [j for j in range(1, len(words) - 1)
                 if is_content_word(words[j])
                 and words.count(words[j]) == 1]
        rng.shuffle(spots)
        for k in spots[:3]:
            blanked = " ".join("____" if j == k else w for j, w in enumerate(words))
            out.append(("rumpang",
                        pool["rumpang"].format(judul=naming(song), lirik=blanked),
                        words[k], "tercetak", ("cloze", sid, i, k)))
    return out


# How many wrong attributions a song is asked to reject, per fact. One was not
# enough: the true claim and its single rival gave two items per fact, which
# left verification twelve times scarcer than the notation categories while the
# test weighted them alike.
#
# Rejecting several different wrong names is also worth more than rejecting one.
# A model shown "not Ismail Marzuki" once can learn that one pairing; shown four
# wrong composers for the same song, the only thing that generalises is the
# right one.
FALSE_CLAIMS = 4


def verify_items(song: dict, pool: dict, others: list[dict],
                 rng: random.Random, extra: dict | None = None) -> list[Item]:
    """True/false statements about the printed metadata.

    Asking the same fact as a claim to be judged, rather than a blank to be
    filled, is what keeps the knowledge from sticking to one question shape.
    The false half also teaches the model to reject a wrong attribution
    instead of agreeing with whatever it is handed.
    """
    out: list[Item] = []
    fields = [("pencipta", "composer", "diciptakan oleh"),
              ("asal", "origin", "berasal dari"),
              ("nada_dasar", "key_signature", "bernada dasar")]
    for name, col, phrase in fields:
        # has(), not truthiness. Everywhere else in this file a blank cell can
        # also be a dash, and the one place that forgot it generated a question
        # whose gold answer was "-".
        if not has(song, col):
            continue
        benar = song[col].strip()
        wordings = askings(pool, extra, "verifikasi")

        for tanya in wordings:
            out.append((f"verifikasi_{name}",
                        tanya.format(judul=naming(song), frasa=phrase,
                                     nilai=benar),
                        f"Benar. Lagu {naming(song)} {phrase} {benar}.",
                        "tercetak", None))

        pilihan = [o[col].strip() for o in others
                   if has(o, col) and o[col].strip() != benar]
        if not pilihan:
            continue

        # Extra false claims are for the TRAINING side only, and the test side
        # keeps its single rival drawn exactly as before. Two reasons, and both
        # matter. A test with four wrong claims per true one makes "Salah" the
        # majority answer, so a model that never agrees with anything scores
        # well. And drawing differently here would shift the random stream that
        # builds the test file, which would silently rewrite a benchmark whose
        # hash is published and whose baseline has already been measured.
        if extra is None:
            salahnya = [rng.choice(pilihan)]
        else:
            unik = sorted(set(pilihan))
            rng.shuffle(unik)
            salahnya = unik[:FALSE_CLAIMS]

        for i, salah in enumerate(salahnya):
            # Each wrong claim gets one wording rather than all of them, so the
            # false half does not outgrow the true half by the size of the
            # paraphrase list. Truth and falsehood stay near enough to even
            # that "Salah" never becomes the safe guess.
            tanya = wordings[i % len(wordings)]
            out.append((f"verifikasi_{name}",
                        tanya.format(judul=naming(song), frasa=phrase,
                                     nilai=salah),
                        f"Salah. Lagu {naming(song)} {phrase} {benar}, "
                        f"bukan {salah}.", "tercetak", None))
    return out


# --- reasoning items ---------------------------------------------------------


def reasoning_items(song: dict, pool: dict, rng: random.Random,
                    limit: int) -> list[Item]:
    """Answers computed from the excerpt itself, so every key is re-derivable.

    Asked per excerpt rather than per song. Per song there would only be four
    reasoning items against thousands of memorisation ones, and the mix would
    be nowhere near the intended share.
    """
    if song["abc_status"] != "terverifikasi":
        return []
    bars = jianpu_bars(song["abc_notation"])
    out: list[Item] = []
    for n in EXCERPT_BARS:
        if len(bars) < n:
            continue
        starts = list(range(0, len(bars) - n + 1))
        rng.shuffle(starts)
        for s in starts[:limit]:
            excerpt = " | ".join(bars[s:s + n])
            pitches = jianpu_pitches(excerpt)
            if not pitches:
                continue
            hi = max(pitches)
            # Deliberately only two kinds. Counting notes or measuring an
            # ambitus is notation arithmetic with nothing Indonesian about it,
            # and the benchmark is about songs, not about counting. What is
            # kept is the smallest control that still works: these are the
            # only questions whose answer sits inside the prompt, so they are
            # the only way a model can score on the 37 songs it never saw.
            # Without them the unseen column is zero by construction and the
            # experiment cannot show generalisation at all.
            out.append(("nada_tertinggi", pool["nada_tertinggi"].format(notasi=excerpt),
                        hi[1], "dihitung", ("excerpt", excerpt)))

    # Counting is asked over its own lengths, not over EXCERPT_BARS. Sharing
    # them meant every answer was 2, 4 or 8 -- all three present in training --
    # so a high score meant telling three memorised lengths apart rather than
    # counting bars, and nothing ever required generalising to 5 or 7. The same
    # defect made the multiple-choice version answerable at 100 percent without
    # reading a note. COUNT_BARS spreads the answers so neither shortcut works.
    panjang = [n for n in COUNT_BARS if len(bars) >= n]
    rng.shuffle(panjang)
    for n in panjang[:limit]:
        s = rng.randrange(0, len(bars) - n + 1)
        excerpt = " | ".join(bars[s:s + n])
        if not jianpu_pitches(excerpt):
            continue
        out.append(("hitung_bar", pool["hitung_bar"].format(notasi=excerpt),
                    str(n), "dihitung", ("excerpt", excerpt)))
    return out


# --- templates ---------------------------------------------------------------

POOL_TRAIN = {
    "pencipta": "Siapa pencipta lagu {judul}?",
    "asal": "Lagu {judul} berasal dari daerah mana?",
    "nada_dasar": "Apa nada dasar lagu {judul}?",
    "tempo": "Bagaimana keterangan tempo lagu {judul}?",
    "jenis": "Lagu {judul} termasuk lagu nasional atau lagu daerah?",
    "birama": "Berapa birama lagu {judul}?",
    "notasi_ke_judul": "Potongan notasi angka ini dari lagu apa?\n{notasi}",
    "lirik_ke_judul": "Potongan lirik ini dari lagu apa?\n{lirik}",
    "lanjut_lirik": "Lanjutkan lirik lagu {judul} setelah baris berikut.\n{lirik}",
    "rumpang": "Lengkapi bagian rumpang pada lirik lagu {judul}.\n{lirik}",
    "verifikasi": "Benarkah lagu {judul} {frasa} {nilai}?",
    "judul_ke_lirik": "Tuliskan lirik lagu {judul}.",
    "judul_ke_baris": "Tuliskan baris ke-{n} lirik lagu {judul}.",
    "judul_ke_notasi": "Tuliskan notasi angka bar {a} sampai {b} lagu {judul}.",
    "hitung_bar": "Ada berapa bar pada notasi angka berikut?\n{notasi}",
    "hitung_not": "Ada berapa not pada notasi angka berikut? Tanda istirahat tidak dihitung.\n{notasi}",
    "ambitus": "Berapa ambitus melodi berikut?\n{notasi}",
    "nada_tertinggi": "Nada apa yang tertinggi pada melodi berikut?\n{notasi}",
    "nada_terendah": "Nada apa yang terendah pada melodi berikut?\n{notasi}",
}

# Deliberately different wording, so the test side measures understanding
# rather than how well the model matched the training phrasing.
POOL_TEST = {
    "pencipta": "Lagu {judul} diciptakan oleh siapa?",
    "asal": "Dari provinsi atau daerah mana asal lagu {judul}?",
    "nada_dasar": "Lagu {judul} memakai nada dasar apa?",
    "tempo": "Tempo lagu {judul} ditulis bagaimana di bukunya?",
    "jenis": "Termasuk kategori apakah lagu {judul}?",
    "birama": "Lagu {judul} bertanda birama berapa?",
    "notasi_ke_judul": "Tentukan judul lagu dari notasi angka berikut.\n{notasi}",
    "lirik_ke_judul": "Sebutkan judul lagu yang memuat baris lirik ini.\n{lirik}",
    "lanjut_lirik": "Baris apa yang menyusul pada lagu {judul}?\n{lirik}",
    "rumpang": "Kata apa yang hilang pada lirik lagu {judul} berikut?\n{lirik}",
    "verifikasi": "Apakah pernyataan ini tepat: lagu {judul} {frasa} {nilai}?",
    "judul_ke_lirik": "Bagaimana bunyi lirik lagu {judul}?",
    "judul_ke_baris": "Baris ke-{n} lirik lagu {judul} berbunyi apa?",
    "judul_ke_notasi": "Bagaimana notasi angka bar {a} hingga {b} lagu {judul}?",
    "hitung_bar": "Hitunglah jumlah bar melodi di bawah ini.\n{notasi}",
    "hitung_not": "Berapa banyak not pada melodi di bawah ini, di luar tanda istirahat?\n{notasi}",
    "ambitus": "Seberapa lebar jangkauan nada melodi di bawah ini?\n{notasi}",
    "nada_tertinggi": "Sebutkan nada paling tinggi pada melodi di bawah ini.\n{notasi}",
    "nada_terendah": "Sebutkan nada paling rendah pada melodi di bawah ini.\n{notasi}",
}

# Extra ways of asking, used on the TRAINING side only.
#
# WHY THIS EXISTS
#
# A song has one composer and one region, so the fact categories yield exactly
# one item per song while the notation categories yield fifty -- a melody can be
# sliced into excerpts, a composer cannot be sliced into anything. With one
# phrasing each, "siapa pencipta lagu X" appeared 55 times in 15,000 training
# examples, 0.37 percent, while notasi_ke_judul took 27 percent. The test then
# weighted them equally, 50 questions apiece. The model was asked to answer
# from something it had been shown three times in three epochs, always in the
# same sentence, and it was measured as though it had been taught.
#
# Repeating one sentence more often would teach the sentence. Asking the same
# fact eight ways teaches the fact, because the only thing constant across the
# eight is the answer. That is also exactly what the test demands: the test
# phrasing is a ninth wording the model has never seen.
#
# Every line here must stay clear of POOL_TEST. main() checks it, and refuses
# to write if a training question ever equals a test question.
PARAPHRASE_TRAIN = {
    "pencipta": [
        "Sebutkan pencipta lagu {judul}.",
        "Siapa yang menggubah lagu {judul}?",
        "Lagu {judul} karya siapa?",
        "Pencipta lagu {judul} adalah siapa?",
        "Tuliskan nama pencipta lagu {judul}.",
        "Menurut buku, siapa pencipta lagu {judul}?",
        "Siapa nama pengarang lagu {judul}?",
    ],
    "asal": [
        "Sebutkan daerah asal lagu {judul}.",
        "Lagu {judul} itu lagu daerah mana?",
        "Asal lagu {judul} dari mana?",
        "Tuliskan asal daerah lagu {judul}.",
        "Daerah mana yang memiliki lagu {judul}?",
        "Menurut buku, lagu {judul} berasal dari mana?",
        "Lagu {judul} datang dari daerah apa?",
    ],
    "nada_dasar": [
        "Sebutkan nada dasar lagu {judul}.",
        "Lagu {judul} dinyanyikan dengan nada dasar apa?",
        "Nada dasar untuk lagu {judul} apa?",
        "Tuliskan nada dasar lagu {judul}.",
        "Do sama dengan apa pada lagu {judul}?",
        "Menurut buku, nada dasar lagu {judul} apa?",
        "Lagu {judul} ditulis dalam nada dasar apa?",
    ],
    "tempo": [
        "Sebutkan tempo lagu {judul}.",
        "Apa tanda tempo lagu {judul}?",
        "Lagu {judul} dibawakan dengan tempo apa?",
        "Tuliskan keterangan tempo lagu {judul}.",
        "Tempo lagu {judul} apa?",
        "Menurut buku, tempo lagu {judul} apa?",
        "Lagu {judul} dinyanyikan dalam tempo apa?",
    ],
    "jenis": [
        "Lagu {judul} tergolong jenis apa?",
        "Sebutkan jenis lagu {judul}.",
        "Apakah {judul} lagu nasional atau lagu daerah?",
        "Tuliskan jenis lagu {judul}.",
        "Lagu {judul} masuk kelompok lagu apa?",
        "Menurut buku, {judul} tergolong lagu apa?",
        "Golongan lagu {judul} apa?",
    ],
    "birama": [
        "Sebutkan birama lagu {judul}.",
        "Apa tanda birama lagu {judul}?",
        "Lagu {judul} memakai birama berapa?",
        "Tuliskan birama lagu {judul}.",
        "Birama lagu {judul} berapa?",
        "Menurut buku, birama lagu {judul} berapa?",
        "Lagu {judul} ditulis dalam birama apa?",
    ],
    "verifikasi": [
        "Apakah benar lagu {judul} {frasa} {nilai}?",
        "Betulkah lagu {judul} {frasa} {nilai}?",
        "Nilailah pernyataan berikut: lagu {judul} {frasa} {nilai}.",
        "Lagu {judul} {frasa} {nilai}, benar atau salah?",
        "Tepatkah kalau dikatakan lagu {judul} {frasa} {nilai}?",
    ],
    "judul_ke_lirik": [
        "Sebutkan lirik lagu {judul}.",
        "Apa lirik lagu {judul}?",
        "Tuliskan seluruh lirik lagu {judul}.",
        "Coba tuliskan lirik lagu {judul}.",
        "Bagaimana teks lirik lagu {judul}?",
        "Tulis ulang lirik lagu {judul}.",
        "Lirik lengkap lagu {judul} seperti apa?",
    ],
}


def askings(pool: dict, extra: dict | None, kategori: str) -> list[str]:
    """Every phrasing for a category: many on the training side, one on the test
    side. Passing extra=None is what makes a build a test build."""
    return [pool[kategori]] + list((extra or {}).get(kategori, []))


MEMORISE = {
    "pencipta", "pencipta_abstain", "asal", "asal_abstain", "nada_dasar",
    "tempo", "tempo_abstain", "jenis", "birama", "notasi_ke_judul", "lirik_ke_judul",
    "lanjut_lirik", "rumpang", "verifikasi_pencipta", "verifikasi_asal",
    "verifikasi_nada_dasar", "judul_ke_lirik", "judul_ke_baris",
    "judul_ke_notasi",
}


# --- assembly ----------------------------------------------------------------


# Categories where the answer is legitimately visible in the question, so the
# giveaway guard below must not fire.
#
# For the notation questions the answer is a note or a count taken FROM the
# melody that is shown: the highest note of a printed melody is necessarily one
# of its notes, and the skill being tested is finding it, not recalling it.
# For "jenis" the question offers both options -- "termasuk lagu nasional atau
# lagu daerah?" -- which is a forced choice, not a leak.
#
# Everything else is filtered: if a question already contains its own answer,
# it measures reading, and a model can score it while knowing no songs.
ANSWER_MAY_APPEAR = {
    "nada_tertinggi", "nada_terendah", "hitung_bar", "hitung_not", "ambitus",
    "judul_ke_notasi", "jenis",
}


def build(songs: list[dict], pool: dict, rng: random.Random, limit: int,
          avoid: frozenset = frozenset(),
          extra: dict | None = None) -> list[dict]:
    """Build one side. Items whose unit is in `avoid` are left out.

    That is the whole mechanism behind the "all" regime: the test side is
    built first and its units become the training side's `avoid` set.
    """
    items: list[dict] = []
    for song in songs:
        raw = (metadata_items(song, pool, extra)
               + verify_items(song, pool, songs, rng, extra)
               + notation_items(song, pool, rng, limit)
               + lyric_items(song, pool, rng, limit)
               + recall_items(song, pool, limit, extra)
               + reasoning_items(song, pool, rng, limit))
        for kategori, tanya, jawab, sumber, unit in raw:
            if not str(jawab).strip():
                continue
            if unit is not None and unit in avoid:
                continue
            if kategori not in ANSWER_MAY_APPEAR and gives_away(str(jawab), tanya):
                continue
            items.append({
                "id_lagu": song["id"],
                "kategori": kategori,
                "tingkat": "hafalan" if kategori in MEMORISE else "penalaran",
                "sumber_kunci": sumber,
                "_unit": unit,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": tanya},
                    {"role": "assistant", "content": str(jawab)},
                ],
            })
    return items


def units_of(items: list[dict]) -> frozenset:
    return frozenset(x["_unit"] for x in items if x["_unit"] is not None)


def dedupe(items: list[dict]) -> list[dict]:
    """Drop repeated question/answer pairs.

    A song that repeats a phrase, or a lyric line that appears in two verses,
    yields the same item twice. Identical copies only inflate the count and
    push the model harder on whatever happened to repeat.
    """
    seen: set[tuple[str, str]] = set()
    out = []
    for x in items:
        key = (x["messages"][1]["content"], x["messages"][2]["content"])
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def sample_to_target(items: list[dict], target: int,
                     rng: random.Random) -> list[dict]:
    """Trim to the target size, spreading the budget evenly across categories.

    WHY THIS REPLACED A MEMORISE/REASON RATIO

    Holding the 90/10 ratio and otherwise sampling at random looked balanced
    and was not, because the pool it sampled from is not balanced. Content
    decides how many items a category can produce: a melody splits into fifty
    excerpts, a composer splits into nothing. Uniform sampling therefore
    reproduced the pool's own skew -- notasi_ke_judul took 27 percent of the
    training set and pencipta took 0.37 percent, a ratio of 75 to 1, while the
    test asked 50 questions of each. The model was trained on one distribution
    and graded on another.

    pick_test twenty lines below had the right rule all along: take the same
    number from every category. This is that rule applied to the side that
    actually decides what the model learns.

    Water-filling, so the budget is not wasted on quotas nobody can fill. The
    scarcest category is served first with everything it has, and whatever it
    could not use is redistributed among the categories that still have items
    left. Categories are served in ascending order of supply, which is what
    makes that a single pass.

    The memorise/reason share is no longer set; it falls out at about 90/10
    because 19 of the 21 categories are memorisation. summarise() prints it so
    a drift away from the intended mixture stays visible.
    """
    # target 0 keeps everything, and that is now the default recipe.
    #
    # Trimming to 15000 was answering a problem that no longer exists. The skew
    # this function was written to fix was 75 to 1, measured before paraphrase
    # augmentation lifted the scarce categories; the pool now runs 4155 down to
    # 440, a ratio of 9.4 to 1. Meanwhile the trim was doing real damage the
    # other way: notasi_ke_judul holds 4155 distinct excerpts and was given
    # 1124, so three quarters of its content was never shown, and that arm
    # scored 6 percent against 82 percent for the untrimmed mixture.
    #
    # Every category covered once beats any allocation rule between them.
    if not target:
        out = list(items)
        rng.shuffle(out)
        return out

    per: dict[str, list[dict]] = {}
    for x in items:
        per.setdefault(x["kategori"], []).append(x)

    out: list[dict] = []
    left = target
    order = sorted(per, key=lambda k: len(per[k]))
    for i, kategori in enumerate(order):
        quota = left // (len(order) - i)
        take = min(len(per[kategori]), quota)
        rng.shuffle(per[kategori])
        out.extend(per[kategori][:take])
        left -= take

    rng.shuffle(out)
    return out


def summarise(name: str, items: list[dict]) -> None:
    print(f"\n{name}: {len(items)} contoh")
    per = Counter(x["kategori"] for x in items)
    for k, v in per.most_common():
        print(f"  {k:22} {v:>6}")
    lvl = Counter(x["tingkat"] for x in items)
    total = sum(lvl.values()) or 1
    for k, v in lvl.most_common():
        print(f"  -- {k:19} {v:>6}  ({v / total * 100:.0f}%)")


def pick_test(items: list[dict], rng: random.Random) -> list[dict]:
    """Even coverage across categories. A benchmark wants spread, not volume."""
    per_cat: dict[str, list[dict]] = {}
    for x in items:
        per_cat.setdefault(x["kategori"], []).append(x)
    out: list[dict] = []
    for k in sorted(per_cat):
        rng.shuffle(per_cat[k])
        out.extend(per_cat[k][:TEST_PER_CATEGORY])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=("all", "split"), default="all",
                    help="all: train on 107 songs, hold out question forms. "
                         "split: keep the frozen 70/37 song split.")
    ap.add_argument("--target", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not SPLIT_PATH.exists():
        print(f"tidak ditemukan: {SPLIT_PATH}", file=sys.stderr)
        print("Jalankan scripts/09_split_songs.py --apply lebih dahulu.", file=sys.stderr)
        return 1

    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    with open(CSV_PATH, encoding="utf-8") as fh:
        rows = {r["id"]: r for r in csv.DictReader(fh)}

    rng = random.Random(args.seed)
    all_songs = list(rows.values())

    # Checked against the templates, not against the sampled output. A wording
    # that collides but happens to be dropped by sampling would pass a check on
    # the files and still be a leak in the generator, waiting for a different
    # seed. The check below on the written items stays as well: this one says
    # the recipe is clean, that one says the batch is.
    bentrok = sorted({t for k, extras in PARAPHRASE_TRAIN.items()
                      for t in extras} & set(POOL_TEST.values()))
    if bentrok:
        print("susunan kalimat latih bertabrakan dengan sisi uji:", file=sys.stderr)
        for t in bentrok:
            print(f"  {t}", file=sys.stderr)
        return 1

    if args.regime == "all":
        train_songs = test_songs = all_songs
        target = args.target or TARGET_ALL
        out_train, out_test = OUT_TRAIN, OUT_TEST
    else:
        train_songs = [rows[i] for i in split["train"]]
        test_songs = [rows[i] for i in split["test"]]
        target = args.target or TARGET_SPLIT
        out_train = OUT_TRAIN.with_name("laguqa_train_split70.jsonl")
        out_test = OUT_TEST.with_name("laguqa_test_split37.jsonl")

    # Test first, so the training side can be told what not to touch.
    test = pick_test(
        dedupe(drop_ambiguous(build(test_songs, POOL_TEST, rng, limit=2), all_songs)),
        rng)
    reserved = units_of(test)

    test, clash_test = drop_clashing(test)

    train = dedupe(drop_ambiguous(
        build(train_songs, POOL_TRAIN, rng, limit=99, avoid=reserved,
              extra=PARAPHRASE_TRAIN), all_songs))
    train, clash_train = drop_clashing(train)
    train = sample_to_target(train, target, rng)

    print(f"{'MODE APPLY' if args.apply else 'MODE DRY-RUN (tidak ada yang ditulis)'}")
    print(f"rezim {args.regime}: lagu latih {len(train_songs)}, "
          f"lagu uji {len(test_songs)}")
    summarise("latih", train)
    summarise("uji", test)

    print(f"\nunit isi yang ditahan untuk sisi uji: {len(reserved)}")

    # Worth reading rather than skimming: a clash is either a lyric the book
    # repeats, which is harmless, or a typo in the dataset, which is not.
    clashes = sorted(set(clash_train) | set(clash_test))
    print(f"pertanyaan berkunci ganda, dibuang: {len(clashes)}")
    for note in clashes[:12]:
        print(f"  {note}")
    if len(clashes) > 12:
        print(f"  ... dan {len(clashes) - 12} lagi")

    gagal = False

    # In the split regime no song may cross. In the all regime every song is
    # meant to cross, and the separation is carried by the units instead.
    if args.regime == "split":
        bocor = {x["id_lagu"] for x in train} & {x["id_lagu"] for x in test}
        print(f"id lagu di kedua sisi: {len(bocor)}"
              + (f"  {sorted(bocor)}" if bocor else "  (bersih)"))
        gagal |= bool(bocor)
    else:
        n_lagu = len({x["id_lagu"] for x in train})
        print(f"lagu yang benar-benar terwakili di sisi latih: {n_lagu}/{len(all_songs)}")
        gagal |= n_lagu != len(all_songs)

    bocor_unit = units_of(train) & reserved
    print(f"unit isi yang bocor ke sisi latih: {len(bocor_unit)}"
          + ("" if bocor_unit else "  (bersih)"))
    gagal |= bool(bocor_unit)

    # Wording alone must never coincide either: an identical question on both
    # sides would be scored as recall of a sentence, not of a song.
    tanya_latih = {x["messages"][1]["content"] for x in train}
    sama = sum(1 for x in test if x["messages"][1]["content"] in tanya_latih)
    print(f"pertanyaan yang sama persis di kedua sisi: {sama}"
          + ("" if sama else "  (bersih)"))
    gagal |= bool(sama)

    if gagal:
        return 1

    if args.apply:
        for path, data in ((out_train, train), (out_test, test)):
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                for x in data:
                    fh.write(json.dumps({k: v for k, v in x.items() if k != "_unit"},
                                        ensure_ascii=False) + "\n")
            print(f"ditulis: {path}  ({len(data)} baris)")
        sha = write_manifest(args.regime, [out_train, out_test],
                             {out_train.name: len(train), out_test.name: len(test)})
        print(f"ditulis: {MANIFEST_PATH.name}  {sha}")
    else:
        print("\nJalankan dengan --apply untuk menulis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
