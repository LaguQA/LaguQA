#!/usr/bin/env python3
"""Pin the two decisions in modal_train.py that fail without saying so.

Both are cheap to get wrong and expensive to notice. Choosing LoRA targets by
name suffix silently attaches adapters to the vision and audio towers, which
never receive a gradient but are still counted as trainable parameters, so the
figure reported in the thesis would be inflated with nothing raising an error.
Splitting a conversation the wrong way silently trains the model on its own
questions, which also runs to completion and also looks fine.

Neither needs a GPU, a download, or torch: both work on module names and on
message lists, which is why they were written to be callable that way.

Usage:
    python tests/test_lora_targets.py
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


def stub_modal() -> None:
    """Stand in for the modal package so the module can be imported offline.

    modal is only needed to submit work to a GPU. Nothing under test touches
    it, so the decorators and builders are replaced by something that accepts
    any call and returns itself.
    """
    class Anything:
        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, name):
            return self

    module = types.ModuleType("modal")
    for name in ("App", "Image", "Volume", "Secret", "Mount"):
        setattr(module, name, Anything())
    sys.modules.setdefault("modal", module)


stub_modal()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modal_train import (  # noqa: E402
    PROJECTIONS, as_prompt_completion, check_targets, select_targets,
)

TEXT_LAYERS = 35
FULL_ATTENTION = 15  # layers 0-14 carry their own keys and values
VISION_LAYERS = 16
AUDIO_LAYERS = 12

# 15 layers with seven projections, 20 with five.
EXPECTED = FULL_ATTENTION * 7 + (TEXT_LAYERS - FULL_ATTENTION) * 5


def gemma4_module_names() -> list[str]:
    """The Linear modules of google/gemma-4-E2B-it, as measured by probe().

    Not a guess. probe() reported two layer shapes: layers 0-14 carry q, k, v
    and o, and layers 15-34 carry only q and o because they reuse the earlier
    layers' keys and values. Every layer also carries the per-layer embedding
    plumbing, which is Linear but is not something to adapt.

    The vision and audio towers here do not name anything q_proj, so the tower
    filter changes nothing for this checkpoint -- which is worth having a test
    say out loud, because it means the guard is currently unexercised by this
    model and is being kept for the other two.
    """
    names = []
    for i in range(TEXT_LAYERS):
        stem = f"model.language_model.layers.{i}"
        attention = ["q_proj", "o_proj"]
        if i < FULL_ATTENTION:
            attention += ["k_proj", "v_proj"]
        names += [f"{stem}.self_attn.{p}" for p in attention]
        names += [f"{stem}.mlp.{p}" for p in ("gate_proj", "up_proj", "down_proj")]
        names += [f"{stem}.per_layer_input_gate", f"{stem}.per_layer_projection"]
    for i in range(VISION_LAYERS):
        stem = f"model.vision_tower.encoder.layers.{i}"
        names += [f"{stem}.attn.qkv", f"{stem}.attn.out", f"{stem}.mlp.fc1"]
    for i in range(AUDIO_LAYERS):
        names += [f"model.audio_tower.layers.{i}.attn.qkv"]
    names += ["model.language_model.per_layer_model_projection", "lm_head"]
    return names


def hostile_module_names() -> list[str]:
    """A checkpoint whose vision tower does use the same projection names.

    Gemma 4 E2B does not, but the other two models have not been probed and the
    guard exists for exactly this case: adapters on a tower that text-only data
    never reaches would train on nothing and still be counted as trainable.
    """
    names = gemma4_module_names()
    for i in range(VISION_LAYERS):
        stem = f"model.vision_tower.encoder.layers.{i}.self_attn"
        names += [f"{stem}.{p}" for p in ("q_proj", "k_proj", "v_proj", "o_proj")]
    return names


RUN = 0


def check(label: str, condition: bool) -> int:
    global RUN
    RUN += 1
    if condition:
        return 0
    print(f"[GAGAL] {label}")
    return 1


def raises(fn) -> bool:
    try:
        fn()
    except RuntimeError:
        return True
    return False


def main() -> int:
    bad = 0
    names = gemma4_module_names()
    targets = select_targets(names)
    shape = check_targets(targets, names)

    bad += check(f"{EXPECTED} target menara teks, dapat {len(targets)}",
                 len(targets) == EXPECTED)
    bad += check("per_layer_* tidak ikut terpilih",
                 not any("per_layer" in n for n in targets))
    bad += check("lm_head tidak ikut terpilih", "lm_head" not in targets)
    bad += check("35 lapis terbaca", shape["layers"] == TEXT_LAYERS)
    bad += check("dua bentuk lapis terbaca, 15 penuh dan 20 berbagi",
                 sorted(shape["shapes"].values()) == [15, 20])

    # The measured checkpoint does not name any vision module q_proj, so the
    # tower filter is a no-op here. Saying so keeps the next reader from
    # believing it was tested when it was not.
    naive = [n for n in names if n.rsplit(".", 1)[-1] in PROJECTIONS]
    bad += check("pada E2B, pemilihan naif kebetulan sama saja",
                 len(naive) == len(targets))

    # On a checkpoint that does share the names, the filter has to bite.
    hostile = hostile_module_names()
    hostile_targets = select_targets(hostile)
    hostile_naive = [n for n in hostile if n.rsplit(".", 1)[-1] in PROJECTIONS]
    bad += check("menara penglihatan bernama sama tetap dibuang",
                 len(hostile_targets) == EXPECTED)
    bad += check(f"pemilihan naif akan menambah {VISION_LAYERS * 4} modul sia-sia",
                 len(hostile_naive) - len(hostile_targets) == VISION_LAYERS * 4)
    bad += check("pemilihan naif ditolak check_targets",
                 raises(lambda: check_targets(hostile_naive, hostile)))

    # A layer stripped of a required projection is a mismatch, not an unusual
    # architecture, and must be refused.
    thin = [n for n in targets if not n.endswith("layers.0.self_attn.q_proj")]
    bad += check("lapis tanpa q_proj ditolak",
                 raises(lambda: check_targets(thin, names)))
    bad += check("target kosong ditolak", raises(lambda: check_targets([], names)))

    # Loss masking: everything up to the last turn is context, the last turn is
    # what the model is scored on learning.
    rows = [{"messages": [{"role": "system", "content": "s"},
                          {"role": "user", "content": "u"},
                          {"role": "assistant", "content": "a"}]}]
    pair = as_prompt_completion(rows)[0]
    bad += check("prompt berisi system dan user saja",
                 [m["role"] for m in pair["prompt"]] == ["system", "user"])
    bad += check("completion berisi giliran asisten saja",
                 [m["role"] for m in pair["completion"]] == ["assistant"])
    bad += check("jawaban tidak bocor ke prompt",
                 all(m["content"] != "a" for m in pair["prompt"]))

    print(f"{RUN - bad}/{RUN} lulus")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
