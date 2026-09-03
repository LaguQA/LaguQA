#!/usr/bin/env python3
"""Cases the lenient scorer has to get right, and traps it must not fall for.

Written as a regression test because every rule in benchmark/evaluate.py is a
judgement call, and a scorer that quietly drifts turns every number in the
thesis into a guess. The golds below are copied from real items in
data/benchmark/laguqa_test.jsonl.

Usage:
    python tests/test_laguqa_eval.py
"""

from __future__ import annotations

import sys

from laguqa.benchmark.evaluate import (REGIME_KEYS, check_alignment, key_order,
                                       load_keys, rival_sets, score)

# (category, gold, prediction, expected lenient verdict, what it checks)
CASES = [
    # Wrapping the answer in a sentence is the normal case, not an error.
    ("pencipta", "Subagyo H.",
     "Lagu Yogya Kembali diciptakan oleh Subagyo H.", True, "answer inside a sentence"),
    ("birama", "4/4", "Biramanya 4/4.", True, "meter inside a sentence"),
    ("nada_dasar", "Do = C", "Nada dasarnya Do = C.", True, "key inside a sentence"),
    ("notasi_ke_judul", "O Ina ni Keke",
     "Notasi itu berasal dari lagu O Ina ni Keke.", True, "title inside a sentence"),
    ("jenis", "Daerah", "Ini termasuk lagu daerah.", True, "case and wrapping"),

    # Hedging by naming several candidates must not score.
    ("pencipta", "Subagyo H.",
     "Bisa Subagyo H., bisa juga Ismail Marzuki.", False, "listing rivals"),
    ("jenis", "Daerah",
     "Bukan lagu nasional, melainkan lagu daerah.", False, "naming the rival"),

    # Key signature is scored on the key named, not on the book's way of
    # writing it. The gold is "Do = C" because that is the jianpu convention
    # the book prints, and only the fine-tuned model ever learns that
    # convention -- it is in the training data. Scoring the convention would
    # make 50 questions a format test the baseline cannot pass and the trained
    # model cannot fail, which is a gap in the thesis that is not knowledge.
    ("nada_dasar", "Do = C", "Nada dasarnya C Mayor.", True, "the other convention"),
    ("nada_dasar", "Do = C", "Lagu ini bernada dasar C.", True, "bare letter in context"),
    ("nada_dasar", "Do = D", "Lagu ini memakai tangga nada D mayor.", True, "scale wording"),
    ("nada_dasar", "Do = C", "Nada dasarnya G Mayor.", False, "wrong key, right format"),
    ("nada_dasar", "Do = C", "Umumnya C Mayor atau G Mayor.", False, "hedging two keys"),
    ("nada_dasar", "Do = C", "do re mi fa sol", False, "solfege is not a key name"),

    # Wrong answers stay wrong.
    ("birama", "4/4", "Biramanya 3/4.", False, "wrong meter"),
    ("hitung_bar", "2", "Ada 3 bar.", False, "wrong count"),

    # Counting: the answer may come first or last.
    ("hitung_bar", "2", "Ada 2 bar.", True, "count stated first"),
    ("hitung_bar", "2", "Bar pertama, lalu bar kedua, jadi 2 bar.",
     True, "count stated after working"),

    # Verification: negation must flip the verdict, not be swallowed.
    ("verifikasi_asal", "Salah. Lagu Bolelebo berasal dari Timor, bukan Bali.",
     "Pernyataan itu kurang tepat.", True, "negated synonym reads as salah"),
    ("verifikasi_asal", "Salah. Lagu Bolelebo berasal dari Timor, bukan Bali.",
     "Tidak benar, asalnya Timor.", True, "'tidak benar' is not 'benar'"),
    ("verifikasi_asal", "Salah. Lagu Bolelebo berasal dari Timor, bukan Bali.",
     "Benar, lagu itu dari Bali.", False, "agreeing with a false claim"),

    # The book heads song 76 "BHINEKA" but sings "Bhin-ne-ka" underneath, so
    # both spellings have to pass and a different title still has to fail.
    ("notasi_ke_judul", "Bhinneka Tunggal Ika",
     "Itu lagu Bhinneka Tunggal Ika.", True, "spelling the dataset uses"),
    ("notasi_ke_judul", "Bhinneka Tunggal Ika",
     "Itu lagu Bhineka Tunggal Ika.", True, "spelling the book's title uses"),
    ("notasi_ke_judul", "Bhinneka Tunggal Ika",
     "Itu lagu Garuda Pancasila.", False, "alias does not loosen everything"),

    # Reasoning models open with their working. It is not the answer, so it is
    # neither credited nor held against them -- weighing Bali out loud before
    # settling on Timor is good reasoning, not hedging.
    ("asal", "Timor",
     "<think>Bisa Bali, bisa Timor. Bukunya menulis Timor.</think>Timor",
     True, "rivals weighed inside think do not count"),
    ("asal", "Timor",
     "<think>Mungkin Timor.</think>Lagu itu dari Bali.",
     False, "answer outside think is the one scored"),
    ("asal", "Timor",
     "<think>Bisa Bali, bisa Timor, kalau dilihat dari liriknya",
     False, "cut off mid-thought means no answer was given"),
    ("pencipta_abstain",
     "Buku sumber tidak mencantumkan pencipta lagu Naik-Naik ke Puncak Gunung.",
     "<think>Sepertinya tidak tercantum.</think>Diciptakan oleh Ibu Sud.",
     False, "abstaining only inside think is not abstaining"),

    # Lyrics: a dropped word is tolerated, a different line is not.
    ("lanjut_lirik", "Indah dan permai bagaikan",
     "Indah dan permai", True, "one word short"),
    ("lanjut_lirik", "Indah dan permai bagaikan",
     "Tanah air beta pusaka abadi", False, "different line"),
    ("judul_ke_baris", "Naik naik ke puncak gunung tinggi tinggi sekali",
     "Naik-naik ke puncak gunung, tinggi-tinggi sekali!",
     True, "hyphens and punctuation"),

    # Abstention: admitting the book is silent is the correct answer.
    ("pencipta_abstain",
     "Buku sumber tidak mencantumkan pencipta lagu Naik-Naik ke Puncak Gunung.",
     "Buku sumber tidak mencantumkan penciptanya.", True, "honest refusal"),
    ("pencipta_abstain",
     "Buku sumber tidak mencantumkan pencipta lagu Naik-Naik ke Puncak Gunung.",
     "Lagu itu diciptakan oleh Ibu Sud.", False, "invented composer"),

    # Declining and then naming someone anyway is the failure these items
    # exist to catch. It used to score full marks: the refusal phrase was
    # matched and the name after it was never looked at.
    ("pencipta_abstain",
     "Buku sumber tidak mencantumkan pencipta lagu Janger.",
     "Buku tidak mencantumkan penciptanya, tetapi kemungkinan besar "
     "ciptaan Ismail Marzuki.", False, "refuses then names a composer anyway"),

    # An untrained model says it does not know rather than citing the book.
    # That is still a refusal to invent, which is the behaviour being tested.
    ("pencipta_abstain",
     "Buku sumber tidak mencantumkan pencipta lagu Janger.",
     "Mohon maaf, saya tidak memiliki informasi spesifik mengenai pencipta "
     "lagu tersebut.", True, "plain admission of ignorance counts"),

    # Confidently naming someone, with no refusal at all, must not pass.
    ("pencipta_abstain",
     "Buku sumber tidak mencantumkan pencipta lagu Janger.",
     "Penciptanya adalah Ismail Marzuki.", False, "no refusal, just a name"),
]


def check_key_files() -> int:
    """The two test sets must not be interchangeable without anyone noticing.

    They share the (id_lagu, kategori) key space but carry different questions,
    so scoring split37 predictions against the full keys pairs answers with
    questions that were never asked. Nothing raises when that happens; the run
    simply reports a wrong accuracy. These checks are the only thing standing
    between that mistake and a number in the thesis.
    """
    bad = 0
    full = load_keys(REGIME_KEYS["full"])
    split = load_keys(REGIME_KEYS["split70"])

    if not REGIME_KEYS["split70"].exists():
        print("[GAGAL] berkas kunci split37 tidak ada")
        return 1

    # The held-out side is a strict subset of songs, and a smaller one.
    songs_full = {sid for sid, _ in full}
    songs_split = {sid for sid, _ in split}
    if not songs_split < songs_full:
        bad += 1
        print(f"[GAGAL] lagu split37 bukan himpunan bagian dari uji penuh: "
              f"{len(songs_split)} vs {len(songs_full)}")

    # And the questions genuinely differ, which is what makes the mix-up
    # silent: if they were identical the wrong key file would be harmless.
    def questions(keys):
        return {x["messages"][1]["content"]
                for items in keys.values() for x in items}
    if questions(full) == questions(split):
        bad += 1
        print("[GAGAL] pertanyaan kedua berkas identik, penebakan kunci tak teruji")

    # The guess predict()'s filenames feed into must land on the right file.
    for stem, want in (("gemma4-e2b-split70-s1--split70", "split70"),
                       ("gemma4-e2b-base--full", "full")):
        regime = stem.rsplit("--", 1)[-1]
        if REGIME_KEYS.get(regime) != REGIME_KEYS[want]:
            bad += 1
            print(f"[GAGAL] {stem} menebak kunci yang salah")
    return bad


def check_ordering() -> int:
    """The scorer must notice predictions that arrived out of order.

    82 slots in the full test set hold more than one question, and the scorer
    tells those apart by position alone. A reordered file is therefore complete,
    correctly formed, fully valid, and scored against the wrong keys. Nothing
    raises. This is the alarm for that.
    """
    bad = 0
    keys = key_order(REGIME_KEYS["full"])

    if check_alignment(keys, keys):
        bad += 1
        print("[GAGAL] urutan yang benar dilaporkan sebagai salah")

    # Sorting by song is the realistic corruption: the test file is written in
    # category blocks, so any regrouping that looks natural moves items across
    # slots. (Sorting by category would be a no-op here, which is the whole
    # reason predict() no longer bothers.)
    by_song = sorted(keys, key=lambda slot: slot[0])
    if by_song == keys:
        bad += 1
        print("[LEWAT] berkas kunci kebetulan sudah urut lagu, uji tak berarti")
    elif not check_alignment(by_song, keys):
        bad += 1
        print("[GAGAL] prediksi teracak tidak terdeteksi")

    # A short file is a different fault with its own report, so this check
    # must stay quiet about it rather than blame the ordering.
    if check_alignment(keys[:-1], keys):
        bad += 1
        print("[GAGAL] prediksi kurang dilaporkan sebagai salah urutan")
    return bad


def main() -> int:
    rivals = rival_sets(load_keys())
    bad = 0
    for kategori, gold, pred, want, note in CASES:
        _, got = score(kategori, pred, gold, rivals[kategori])
        if got != want:
            bad += 1
            print(f"[GAGAL] {kategori:22} {note}")
            print(f"         gold: {gold[:60]}")
            print(f"         pred: {pred[:60]}")
            print(f"         expected {want}, got {got}")
    print(f"{len(CASES)} cases, {bad} failing")

    key_bad = check_key_files()
    print(f"pemeriksaan berkas kunci: {'lulus' if not key_bad else f'{key_bad} gagal'}")

    order_bad = check_ordering()
    print(f"pemeriksaan urutan prediksi: {'lulus' if not order_bad else f'{order_bad} gagal'}")
    return 1 if bad or key_bad or order_bad else 0


if __name__ == "__main__":
    sys.exit(main())
