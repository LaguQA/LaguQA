#!/usr/bin/env python3
"""modal_train.mc_messages must equal multichoice.mc_messages, item for item.

The prompt is written twice on purpose -- once locally, once inside the file
that ships to the GPU -- because the container has no laguqa package. Two
copies drift, and a drifted prompt evaluates one question and scores another
with no error anywhere. This is what stops that.

Usage:
    python tests/test_laguqa_mc_prompt.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal_train
from laguqa.benchmark.multichoice import PROMPT_SISTEM, mc_messages, read_mc

# Every multiple-choice file, not just LaguQA's. The forgetting probes go
# through the same two functions, and an item shape that renders differently on
# one side of the copy would be found here rather than after a paid run.
BERKAS = [
    Path("data/benchmark/laguqa_mc.jsonl"),
    Path("data/eksternal/indommlu_mc.jsonl"),
    Path("data/eksternal/indoculture_mc.jsonl"),
]


def main() -> int:
    if PROMPT_SISTEM != modal_train.PROMPT_SISTEM:
        # Checked before the items, because the whole forgetting comparison
        # rests on the base model and the fine-tuned model having been asked
        # under the same two system messages. If these tables ever diverge, the
        # two conditions stop being conditions and become two different
        # experiments with matching file names.
        print("GAGAL: tabel PROMPT_SISTEM berbeda")
        print(f"  multichoice: {PROMPT_SISTEM}")
        print(f"  modal_train: {modal_train.PROMPT_SISTEM}")
        return 1

    galat = 0
    for path in BERKAS:
        if not path.is_file():
            print(f"lewat: {path} tidak ada")
            continue
        items = read_mc(path)
        for nama, sistem in sorted(PROMPT_SISTEM.items()):
            beda = [x["id"] for x in items
                    if modal_train.mc_messages(x, sistem) != mc_messages(x, sistem)]
            print(f"{path.name:24} prompt {nama:7} "
                  f"{len(items):5} soal, {len(beda)} berbeda")
            if beda:
                galat += 1
                print(f"  GAGAL contoh: {beda[:3]}")
                a = modal_train.mc_messages(items[0], sistem)
                b = mc_messages(items[0], sistem)
                print(f"  modal_train: {a!r}\n  multichoice: {b!r}")
    return 1 if galat else 0


if __name__ == "__main__":
    sys.exit(main())
