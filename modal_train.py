#!/usr/bin/env python3
"""Fine-tune a base model on LaguQA with LoRA, on Modal.

Three base models are trained, chosen so the comparison answers a question
instead of ranking vendors. Every figure below was measured by probe(), not
copied from a model card:

    name           model_id                                 params      layers  modules   trainable
    gemma4-e2b     google/gemma-4-E2B-it                5,104,297,504   15+20     205    24,158,208
    sealion-e2b    aisingapore/Gemma-SEA-LION-v4.5-...  5,104,297,504   15+20     205    24,158,208
    gemma4-e4b     google/gemma-4-E4B-it                7,941,100,832   24+18     258    34,881,536

The first two rows are identical in every structural figure. SEA-LION v4.5 E2B
declares gemma-4-E2B-it as its own base model, and the only difference is
continued pretraining on Southeast Asian languages, Indonesian among them. So
gemma4-e2b against sealion-e2b isolates one variable, language, and gemma4-e2b
against gemma4-e4b isolates the other, size. Any other pairing would confound
the two and could answer neither question.

All three are Apache-2.0 or MIT and none is gated, so an examiner can download
them without a token and without agreeing to anything.

WHAT probe() CORRECTED

"15+20" is the correction. The layers are not built alike: the first 15 carry
their own q, k, v and o, and the remaining 20 carry only q and o because they
reuse the earlier layers' keys and values. An earlier version of check_targets
demanded seven projections per layer, computed 245, and refused to start. The
checkpoint was right and the assumption was wrong.

"15 + 20" is the correction. The layers are not built alike: the first 15 carry
their own q, k, v and o, and the remaining 20 carry only q and o because they
reuse the earlier layers' keys and values. An earlier version of check_targets
demanded seven projections per layer, computed 245, and refused to start. The
checkpoint was right and the assumption was wrong.

The tower filter, on the other hand, turned out not to be needed here, and
saying so matters more than claiming a save. Each checkpoint wraps three
towers -- text, vision, audio -- but the vision and audio towers name their
projections input_proj, output_proj and relative_k_proj, so PEFT's own matching
already misses them: it looks for ".k_proj" with the dot in front. Drop the
dot, as a hand-written regex or an older PEFT does, and "relative_k_proj"
matches. That is 12 modules, one per audio layer, that would receive no
gradient from text-only data and still be counted as trained. probe() reports
both numbers so the margin is measured rather than argued about, and
select_targets() keeps the filter because no later model is promised to be as
lucky.

The last oddity is the name. "E2B" means 2.3B effective parameters but 5.1B
stored; the difference is the Per-Layer Embedding tables, which are large
lookups rather than compute. Memory follows the stored count -- 9.6 GB resident
for E2B, 14.9 GB for E4B, 12.5 GB peak while training E2B -- so the GPU is
sized against 5.1B, not 2.3B.

Usage:
    pip install modal && modal token new      # once
    modal run modal_train.py::upload          # push the benchmark to a volume
    modal run modal_train.py::inspect_api --wanted a,b,c   # CPU: API check
    modal run modal_train.py::probe           # cheap: does LoRA attach cleanly?
    modal run modal_train.py::smoke           # cheap: does loss go down?
    modal run modal_train.py::bench           # cheap: which GPU is cheapest?
    modal run modal_train.py::run --model gemma4-e2b --seed 1
    modal run modal_train.py::predict --model gemma4-e2b --run-id <id>
    modal run modal_train.py::experiment      # the whole grid, in parallel
    modal run modal_train.py::fetch           # pull manifests and predictions
    modal run modal_train.py::release --run-id <id>   # build a Hub folder

Then, locally and with no GPU:

    python scripts/17_controls.py             # what does knowing nothing score?
    python scripts/11_evaluate.py hasil/<file>.jsonl
    python scripts/16_figures.py              # figures plus the CSVs behind them
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import modal

# --- what gets trained -------------------------------------------------------

MODELS = {
    "gemma4-e2b": "google/gemma-4-E2B-it",
    "sealion-e2b": "aisingapore/Gemma-SEA-LION-v4.5-E2B-IT",
    "gemma4-e4b": "google/gemma-4-E4B-it",
}

# Measured zero-shot and never fine-tuned. These say how much a capable model
# already knows about the book's songs before anyone trains on it, which is the
# number that gives the trained results their meaning: without it, an accuracy
# of 60 percent could be teaching or could be what the model walked in with.
#
# Ornith is here rather than in MODELS deliberately. It is a strong model, but
# it is a reasoning model whose assistant turn opens with a <think> block,
# while every answer in the training data is bare. Fine-tuning it on 10,000
# such answers would train that habit out, which both wastes what makes it good
# and confounds the comparison, since the Gemma models are not reasoning models
# and would not be damaged the same way. As a baseline none of that applies.
#
# WHY THESE AND NOT OLDER, BETTER-KNOWN ONES
#
# An examiner asked for at least ten models. The temptation is to pad the list
# with names that are easy to recall -- Merak, Cendol, Sailor2, SeaLLMs,
# Llama 3.2 -- but those are 2023-2024 releases, and putting a 2026 Gemma 4
# against them measures two years of general progress rather than knowledge of
# this songbook. Every entry below was checked against the Hub in September
# 2026: open weights, not gated, safetensors, and its own chat template.
#
# Two candidates were dropped for reasons worth keeping:
#   - Cendol ships no chat template. Writing one by hand would give a single
#     model a prompt format invented here while the rest use their own, which
#     is the same asymmetry that made the 256-token ceiling misleading.
#   - Llama 3.2 and CohereLabs tiny-aya are gated behind manual approval, so a
#     third party could not reproduce the run.
#
# SahabatAI is the exception on age, deliberately. It is the most-downloaded
# Indonesian-specific instruct model and nothing newer exists at a size that
# fits: v2 was released only at 70B, and Nusantara and Komodo stopped in 2024.
# Dropping it would remove Indonesian-specific models from the comparison
# entirely. That the newest usable one dates to 2025 is itself reportable.
BASELINES = {
    "ornith-9b": "ornith-ai/Ornith-1.5-9B",
    "qwen-sealion-4b": "aisingapore/Qwen-SEA-LION-v4-4B-VL",
    # multibahasa umum, 2026
    "qwen35-4b": "Qwen/Qwen3.5-4B",
    "qwen35-9b": "Qwen/Qwen3.5-9B",
    "granite42-8b": "ibm-granite/granite-4.2-8b",
    "lfm25-2b": "LiquidAI/LFM2.5-2.6B",
    "smollm3-3b": "HuggingFaceTB/SmolLM3-3B",
    # Asia Tenggara
    "apertus-sealion-8b": "aisingapore/Apertus-SEA-LION-v4-8B-IT",
    "sealion-v35-8b": "aisingapore/Llama-SEA-LION-v3.5-8B-R",
    # khusus Indonesia
    "sahabatai-9b": "Sahabat-AI/gemma2-9b-cpt-sahabatai-v1-instruct",
}

# Models that need more room than the default before their answer even starts.
# A reasoning model spends its first few hundred tokens inside <think>, and the
# scorer strips that block entirely; if generation stops before the block
# closes, there is no answer left to score and the model records a zero it did
# not earn. This is the same truncation trap that was cutting the Gemma
# baseline at 96 tokens, except it fails to nothing instead of to a fragment.
#
# The reasoning models below get 2048 rather than 1024. On multiple choice the
# answer is one letter, so a budget looks absurdly generous -- but the letter
# comes *after* the think block, and a run that stops mid-thought scores zero
# without the model ever having been wrong. The truncation rate is printed per
# run; if it is not zero for these, the budget is still the thing being
# measured and must be raised again before the number is used.
TOKEN_BUDGET = {
    "ornith-9b": 1024,
    "sealion-v35-8b": 2048,
    "smollm3-3b": 2048,
    "qwen35-4b": 2048,
    "qwen35-9b": 2048,
}


def budget_for(model: str, asked: int) -> int:
    return max(asked, TOKEN_BUDGET.get(model, 0))


def resolve(model: str) -> str:
    if model in MODELS:
        return MODELS[model]
    if model in BASELINES:
        return BASELINES[model]
    raise KeyError(f"model tidak dikenal: {model}. "
                   f"pilih dari {sorted(MODELS) + sorted(BASELINES)}")

# The four training sets. They answer two different questions and their numbers
# are not comparable to each other.
#
#   split70/split37 holds out 37 SONGS. Nothing about those songs was ever
#   shown, so this measures skill that transfers: reading notation, counting
#   bars. It cannot measure knowledge injection, by construction -- no amount
#   of training teaches lyrics of a song that was removed from the data, and
#   the lyric categories sit at 0% for base and trained alike.
#
#   full/laguqa_test trains on all 107 songs and holds out question FORMS: zero
#   test questions appear verbatim in the training set (checked, both regimes).
#   This is the knowledge-injection measurement -- can the facts be put into the
#   weights and come back out when asked differently?
#
# Both are legitimate; conflating them is not. A full-regime number is
# memorisation reached through an unseen phrasing, and reporting it as
# generalisation to new songs would be a lie. That claim belongs to split70 and
# only to split70.
#   mc/laguqa_mc is EVALUATION ONLY. It is the headline track the proposal was
#   approved on -- five options, scored by letter equality -- and it has no
#   training half by design: training on it would teach the answer key of the
#   thing being measured. run() refuses it for that reason.
#
#   indommlu/indoculture are the forgetting probes, evaluation only and never
#   about songs. They exist because every other number here comes from the one
#   domain the adapter was trained on, so nothing else in this file can tell a
#   model that gained Indonesian songs from one that gained Indonesian songs and
#   lost its chemistry. Built by scripts/26_external_bench.py; the sources are
#   CC BY-NC-SA 4.0, which is why they sit in data/eksternal/ and not alongside
#   the LaguQA files. run() refuses them for the same reason it refuses mc.
#   full14 is the v1.4 build of the full regime, the one the released root
#   adapter learned from. It exists so that a repeat of that run is possible at
#   all: the original files were overwritten in place by the v1.5 rebuild and no
#   copy survived on either volume. Rebuilt byte-identically by
#   scripts/30_rebuild_v14.py, and carrying its own file names so that upload()
#   cannot repeat the overwrite. Use it only to reproduce or extend v1.4 runs;
#   v1.5 is the current benchmark and full is still the default.
DATASETS = {
    "split70": ("laguqa_train_split70.jsonl", "laguqa_test_split37.jsonl"),
    "full": ("laguqa_train.jsonl", "laguqa_test.jsonl"),
    "full14": ("laguqa_train_v14.jsonl", "laguqa_test_v14.jsonl"),
    "mc": (None, "laguqa_mc.jsonl"),
    "indommlu": (None, "indommlu_mc.jsonl"),
    "indoculture": (None, "indoculture_mc.jsonl"),
}

# A dataset with no training half is a multiple-choice file: questions and
# options rather than chat turns, so predict() has to build the prompt itself.
# Derived from the table instead of listed by name, because the previous test
# was `dataset == "mc"` and every benchmark added after it would have been fed
# to the model as raw JSON without one error being raised.
MC_DATASETS = {k for k, (train, _) in DATASETS.items() if train is None}

# Modal's published per-hour rates, read 2026-09-01. Used only to record an
# estimated cost per run; the invoice is the authority.
GPU_RATE = {"L4": 0.80, "A10G": 1.10, "L40S": 1.95, "A100": 2.10, "H100": 3.95}

# Measured by bench(), 30 identical steps on gemma4-e2b, all at 12.5 GB peak:
#
#     card      s/step   full run   usd/run
#     L4          3.82     1.99 h      1.59
#     A100        2.77     1.44 h      3.03
#     L40S        1.38     0.72 h      1.40
#
# L40S is both the fastest and the cheapest, which is not something you can
# read off a price list: it is 2.8x quicker than an L4 for 2.4x the hourly
# rate. The A100 was the worst of the three on both counts. (Its figure is
# approximate -- the scheduler handed back an 80GB card for a 40GB request,
# which bills higher than the rate above.)
#
# Overridable because the two Modal accounts funding this are not offered the
# same hardware. A workspace with no billing method on file is refused L40S,
# A100 and H100 outright -- the error arrives when the app is created, so the
# card is not a preference there, it is a precondition. T4, L4 and A10G are
# allowed, and an L4 fits this training at 12.5 GB peak against 23 GB, so the
# second account is usable for real runs rather than only for trials. It is
# slower per step and, at $0.80 an hour, still cheaper per run.
#
# The manifest records the card the driver reported, not the one asked for, so
# changing this cannot quietly mislabel a result.
GPU = os.environ.get("LAGUQA_GPU", "L40S")


def rate_for(label: str) -> float:
    """Hourly rate for whatever card the container was actually given.

    Matched against the name the driver reports rather than the name requested,
    so a run that landed on different hardware is costed as what it ran on.
    Longest key first, or "L4" would swallow "L40S".
    """
    upper = label.upper()
    for key in sorted(GPU_RATE, key=len, reverse=True):
        if key in upper:
            return GPU_RATE[key]
    return 0.0

# Four sequences per step, accumulated four times, so sixteen examples inform
# each update. Sized so E4B's 14.9 GB of resident weights still leaves room for
# activations; E2B peaks at 12.5 GB this way, well inside an L40S.
BATCH = 4
ACCUM = 4
WARMUP_FRACTION = 0.03

app = modal.App("laguqa")

# Pinned to what the first probe and smoke runs actually resolved to, so the
# pin describes runs that happened rather than versions someone hoped for.
# These are newer than most LoRA tutorials and the difference is not cosmetic:
# trl 1.12 removed warmup_ratio in favour of warmup_steps, which cost one
# failed run to discover. inspect_api() below reports what a rebuilt image
# accepts, which is the cheap way to find the next such change.
# Copied from laguqa.benchmark.multichoice rather than imported: the image
# ships this file alone, and pulling the package in drags CSV paths and the
# jianpu helpers into a GPU container that has no use for them. The copy is
# held to the original by tests/test_laguqa_mc_prompt.py, which fails if the
# two ever produce different text -- a silent divergence here would evaluate
# one prompt and score another.
MC_SYSTEM = (
    "Kamu asisten yang menguasai lagu nasional dan lagu daerah Indonesia, "
    "termasuk notasi angka dan notasi ABC-nya."
)
MC_INSTRUKSI = "Jawab hanya dengan satu huruf pilihan."
MC_SISTEM_NETRAL = "Kamu asisten yang menjawab soal pilihan ganda."

PROMPT_SISTEM = {"lagu": MC_SYSTEM, "netral": MC_SISTEM_NETRAL}


def mc_messages(item: dict, sistem: str = MC_SYSTEM) -> list[dict]:
    opsi = "\n".join(f"{k}. {v}" for k, v in sorted(item["opsi"].items()))
    return [
        {"role": "system", "content": sistem},
        {"role": "user",
         "content": f"{item['pertanyaan']}\n\n{opsi}\n\n{MC_INSTRUKSI}"},
    ]


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.16.1",
        "peft==0.20.0",
        "trl==1.12.0",
        "datasets==5.0.1",
        "accelerate==1.14.0",
    )
    .env({"HF_HOME": "/cache", "TOKENIZERS_PARALLELISM": "false"})
)

cache = modal.Volume.from_name("laguqa-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("laguqa-data", create_if_missing=True)
runs = modal.Volume.from_name("laguqa-runs", create_if_missing=True)
VOLUMES = {"/cache": cache, "/data": data_vol, "/runs": runs}

HERE = Path(__file__).resolve().parent
BENCHMARK_DIR = HERE / "data" / "benchmark"
# Separate directory, separate licence. The external probes are CC BY-NC-SA 4.0
# derivatives and must stay clear of the CC BY 4.0 dataset release, which picks
# its payload up with BENCHMARK_DIR.glob("*.jsonl").
EXTERNAL_DIR = HERE / "data" / "eksternal"
# v1.4, rebuilt by scripts/30_rebuild_v14.py. Its own directory for the same
# reason as the line above, and under its own file names so that upload() cannot
# overwrite v1.5 with it. Overwriting under a shared name is exactly how the
# original v1.4 files stopped existing.
ARCHIVE_DIR = HERE / "data-v14" / "benchmark"

ROOTS = (BENCHMARK_DIR, EXTERNAL_DIR, ARCHIVE_DIR)


def local_dataset(name: str) -> Path:
    """Where a dataset file lives locally, searched across every root."""
    for root in ROOTS:
        if (root / name).exists():
            return root / name
    raise SystemExit(f"tidak ada: {name} di "
                     + ", ".join(str(r) for r in ROOTS))


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- pieces that run inside the container ------------------------------------


def load_base(model_id: str):
    """Load the checkpoint, whichever auto class it happens to be registered under.

    The three models are tagged any-to-any on the Hub, and transformers maps
    that family to different auto classes across releases. Trying them in order
    is uglier than naming one, but it is the difference between a run that
    survives a transformers upgrade and one that dies at 3am.
    """
    import torch
    import transformers

    names = ("AutoModelForCausalLM", "AutoModelForImageTextToText",
             "AutoModelForPreTraining")
    last: Exception | None = None
    for name in names:
        cls = getattr(transformers, name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(
                model_id, dtype=torch.bfloat16, device_map="cuda:0",
                attn_implementation="eager",
            )
            print(f"dimuat lewat {name}: {type(model).__name__}")
            return model, name
        except Exception as exc:  # noqa: BLE001 - the next class may work
            last = exc
            print(f"{name} gagal: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"tidak ada auto class yang bisa memuat {model_id}") from last


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj")

# Substrings that mark a module as belonging to a tower we do not train.
OTHER_TOWERS = ("vision", "audio", "multi_modal_projector", "mm_projector")


def select_targets(names) -> list[str]:
    """Keep the text tower's projections and drop everything else.

    Full names rather than suffixes on purpose. PEFT matches a bare "q_proj"
    against every module ending that way, which on these checkpoints includes
    the vision and audio towers. Those adapters would train on nothing and
    still be counted, so the parameter count reported in the thesis would be
    inflated by a mistake that raises no error.

    Kept separate from the model so it can be tested without a GPU, or a
    download, or torch: it takes module names and returns module names.
    """
    return [n for n in names
            if n.rsplit(".", 1)[-1] in PROJECTIONS
            and not any(t in n for t in OTHER_TOWERS)]


# What every decoder layer must contribute, whatever else it has. A layer
# missing one of these has been matched wrongly, not built unusually.
REQUIRED = ("q_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def check_targets(targets: list[str], names: list[str]) -> dict:
    """Refuse a target list that does not match the checkpoint in front of it.

    An earlier version of this demanded seven projections per layer and
    rejected a correct selection. Gemma 4 E2B shares keys and values: its first
    15 layers carry q, k, v and o, and the remaining 20 carry only q and o
    because they reuse the earlier layers' keys and values. 15x7 + 20x5 = 205,
    not 245. The architecture was right and the assumption was wrong.

    So the rule is weaker and true instead of neat: nothing from another tower
    got in, every text layer is represented, each one keeps at least its query,
    output and three MLP projections, and no projection that does exist was
    left behind. That still catches the failure this guard was written for --
    adapters attached to a tower that never sees a gradient -- without
    inventing a shape the model does not have.
    """
    stray = [n for n in targets if any(t in n for t in OTHER_TOWERS)]
    if stray:
        raise RuntimeError(f"{len(stray)} modul dari menara lain ikut terpilih: "
                           f"{stray[:3]}")

    missed = sorted(set(select_targets(names)) - set(targets))
    if missed:
        raise RuntimeError(f"{len(missed)} proyeksi menara teks tertinggal: "
                           f"{missed[:3]}")

    per_layer: dict[int, set[str]] = {}
    for name in targets:
        if ".layers." not in name:
            continue
        index = int(name.split(".layers.")[1].split(".")[0])
        per_layer.setdefault(index, set()).add(name.rsplit(".", 1)[-1])
    if not per_layer:
        raise RuntimeError("tidak ada satu lapis pun yang terpilih")

    thin = {i: sorted(s) for i, s in per_layer.items() if not set(REQUIRED) <= s}
    if thin:
        first = sorted(thin)[0]
        raise RuntimeError(f"{len(thin)} lapis kekurangan proyeksi wajib, "
                           f"misalnya lapis {first}: {thin[first]}")

    shapes: dict[str, int] = {}
    for found in per_layer.values():
        key = ",".join(sorted(found))
        shapes[key] = shapes.get(key, 0) + 1
    return {"layers": len(per_layer), "modules": len(targets), "shapes": shapes}


def linear_names(model) -> list[str]:
    import torch

    return [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def as_prompt_completion(rows: list[dict]) -> list[dict]:
    """Split each conversation so the loss lands only on the assistant turn.

    This is the rule the whole training design rests on. The facts to be
    memorised live in the answer, so the answer is what the loss has to see;
    if the question were scored too, the model would spend its capacity
    learning to reproduce prompts it will be given anyway. TRL masks the
    prompt automatically for data in this shape, which is why the shape is
    chosen here rather than relying on a chat template carrying generation
    markers -- these templates do not.
    """
    return [{"prompt": r["messages"][:-1], "completion": r["messages"][-1:]}
            for r in rows]


# Share of the training file held back to watch for overfitting. Carved from
# TRAINING data, never from the test side: the 37 held-out songs have to stay
# untouched or the final accuracy stops meaning what it claims. This slice
# holds items about songs the model does train on, so a rising loss here is
# the model memorising particular items rather than the songs behind them --
# which is exactly the failure three epochs might cause and a loss curve on
# training data alone would hide.
VALIDATION_SHARE = 0.05


def carve_validation(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """Split off a validation slice, deterministically for a given seed."""
    import random as _random

    shuffled = list(rows)
    _random.Random(seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * VALIDATION_SHARE))
    return shuffled[cut:], shuffled[:cut]


def versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {}
    for name in ("torch", "transformers", "peft", "trl", "datasets", "accelerate"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = "not installed"
    return out


# --- Modal functions ---------------------------------------------------------


@app.function(image=image, timeout=60 * 10)
def inspect_api(wanted: str = "") -> dict:
    """Ask the image which SFTConfig settings it actually accepts.

    The image pins floors, not versions, so it resolved to trl 1.12 and
    transformers 5.16 -- far newer than the tutorials this recipe was written
    from, and TRL 1.x renamed and removed settings. Guessing costs a GPU
    minute per wrong guess. This runs on CPU, costs almost nothing, and answers
    the question directly.
    """
    import dataclasses

    from trl import SFTConfig

    fields = sorted(f.name for f in dataclasses.fields(SFTConfig))
    names = [w.strip() for w in wanted.split(",") if w.strip()]
    report = {
        "versions": versions(),
        "n_fields": len(fields),
        "missing": [n for n in names if n not in fields],
        "present": [n for n in names if n in fields],
        "warmup_like": [f for f in fields if "warmup" in f or "schedul" in f],
        "loss_like": [f for f in fields if "loss" in f or "completion" in f],
        "length_like": [f for f in fields if "length" in f or "max_seq" in f],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=60 * 30)
def probe(model: str = "gemma4-e2b") -> dict:
    """Answer the one question that decides whether this plan works at all.

    Loads the checkpoint, lists what is inside it, and reports exactly which
    modules LoRA would attach to. No training, no data, a few minutes on the
    cheapest GPU. If the text tower cannot be told apart from the vision and
    audio towers, that is worth finding out here rather than after paying for
    nine training runs.
    """
    import torch
    from peft import LoraConfig, get_peft_model

    model_id = resolve(model)
    base, auto_class = load_base(model_id)

    total = sum(p.numel() for p in base.parameters())

    # One level down as well as at the top: the towers live inside a wrapper,
    # so the top level is a single "model" holding everything and says nothing.
    towers: dict[str, int] = {}
    for prefix, holder in (("", base), ("model.", getattr(base, "model", None))):
        if holder is None:
            continue
        for name, child in holder.named_children():
            towers[prefix + name] = sum(p.numel() for p in child.parameters())

    names = linear_names(base)
    targets = select_targets(names)

    # Two ways of matching by suffix, measured rather than assumed. PEFT keeps
    # a dot in front of the name it looks for, so ".k_proj" does not match the
    # audio tower's "relative_k_proj". Drop the dot -- which a hand-written
    # regex or an older PEFT does -- and it does. The gap between these two
    # numbers is how close the usual approach comes to adapting a tower that
    # text-only data never reaches.
    naive = [n for n in names if any(n.endswith("." + p) for p in PROJECTIONS)]
    loose = [n for n in names if any(n.endswith(p) for p in PROJECTIONS)]

    # Every distinct Linear name in the text tower, and how the projections are
    # distributed over its layers. A checkpoint whose layers are not all built
    # the same -- shared attention, a mixture of experts, fused qkv -- shows up
    # here as more than one shape, which is the thing worth knowing before
    # choosing what to adapt.
    text = [n for n in names if not any(t in n for t in OTHER_TOWERS)]
    text_kinds = sorted({n.rsplit(".", 1)[-1] for n in text})
    other_kinds = sorted({n.rsplit(".", 1)[-1] for n in names if n not in set(text)})
    per_layer: dict[int, list[str]] = {}
    for name in text:
        if ".layers." not in name:
            continue
        index = int(name.split(".layers.")[1].split(".")[0])
        per_layer.setdefault(index, []).append(name.rsplit(".", 1)[-1])
    grouped: dict[str, list[int]] = {}
    for index, found in per_layer.items():
        grouped.setdefault(",".join(sorted(found)), []).append(index)
    layer_shapes = {k: {"layers": len(v), "first": sorted(v)[:3]}
                    for k, v in grouped.items()}

    try:
        summary = check_targets(targets, names)
        guard = "lolos"
    except RuntimeError as exc:
        # probe exists to describe the checkpoint, so it reports the mismatch
        # instead of aborting on it. _train still refuses to start.
        print(f"CATATAN: {exc}")
        summary = {"layers": len(per_layer), "modules": len(targets)}
        guard = str(exc)

    peft_model = get_peft_model(base, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=targets,
    ))
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)

    report = {
        "model": model,
        "model_id": model_id,
        "auto_class": auto_class,
        "model_class": type(base).__name__,
        "total_params": total,
        "towers": towers,
        "text_layers": summary["layers"],
        "guard": guard,
        "layer_shapes": layer_shapes,
        "text_linear_kinds": text_kinds,
        "other_tower_linear_kinds": other_kinds,
        "lora_targets": len(targets),
        "naive_dotted_suffix": len(naive),
        "naive_loose_suffix": len(loose),
        "wasted_if_dotted": len(naive) - len(targets),
        "wasted_if_loose": len(loose) - len(targets),
        "caught_only_by_loose": sorted({n.rsplit(".", 1)[-1]
                                        for n in set(loose) - set(targets)}),
        "trainable_params": trainable,
        "trainable_percent": round(100 * trainable / total, 4),
        "target_sample": targets[:3],
        "versions": versions(),
        "gpu_free_gb": round(torch.cuda.mem_get_info()[0] / 2**30, 1),
        "gpu_used_gb": round(torch.cuda.memory_allocated() / 2**30, 1),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


@app.function(image=image, gpu=GPU, timeout=60 * 5)
def mesin() -> dict:
    """Report what a GPU container on this workspace actually gets.

    Bab III has to state the CPU, memory and card the experiments ran on, and
    none of that is written anywhere in this file: the GPU is chosen by name,
    everything else is whatever Modal allocates alongside it. Guessing the
    numbers from documentation would put unverified figures in a thesis, so
    they are measured here instead. Cheap enough to be free in practice --
    a few seconds of one card.

        modal run --detach modal_train.py::mesin
    """
    import os
    import torch

    def meminfo(kunci: str) -> float:
        try:
            for baris in Path("/proc/meminfo").read_text().splitlines():
                if baris.startswith(kunci):
                    return round(int(baris.split()[1]) / 2**20, 1)
        except OSError:
            pass
        return 0.0

    def cgroup(*jalur: str) -> str:
        for p in jalur:
            try:
                return Path(p).read_text().strip()
            except OSError:
                continue
        return ""

    model_cpu = ""
    try:
        for baris in Path("/proc/cpuinfo").read_text().splitlines():
            if baris.startswith("model name"):
                model_cpu = baris.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    props = torch.cuda.get_device_properties(0)
    laporan = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_gb": round(props.total_memory / 2**30, 1),
        "gpu_capability": f"{props.major}.{props.minor}",
        "gpu_multiprocessors": props.multi_processor_count,
        "cpu_model": model_cpu,
        "cpu_terlihat": os.cpu_count(),
        "cpu_terpakai": len(os.sched_getaffinity(0)),
        "cpu_quota_cgroup": cgroup("/sys/fs/cgroup/cpu.max"),
        "ram_total_gb": meminfo("MemTotal"),
        "ram_tersedia_gb": meminfo("MemAvailable"),
        "ram_batas_cgroup": cgroup("/sys/fs/cgroup/memory.max"),
        "kartu_per_wadah": torch.cuda.device_count(),
        "driver_cuda": torch.version.cuda,
        "versions": versions(),
    }
    print(json.dumps(laporan, indent=2, ensure_ascii=False))
    return laporan


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=60 * 60)
def smoke(model: str = "gemma4-e2b", steps: int = 30) -> dict:
    """Train for a handful of steps to prove the loss actually moves.

    probe() shows the adapter attaches. This shows it learns: same code path as
    a real run, 30 steps on a slice of the data, under a dollar. If the loss is
    flat here it will be flat after five epochs too, and the reason will be far
    harder to find once nine runs are in flight.
    """
    return _train(model=model, dataset="split70", seed=0, epochs=1,
                  limit=256, max_steps=steps, tag="smoke")


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=60 * 60 * 4)
def run(model: str = "gemma4-e2b", dataset: str = "split70", seed: int = 1,
        epochs: int = 3, tag: str = "run", lora_r: int = 16,
        lora_alpha: int = 0, lr: float = 2e-4) -> dict:
    """One real training run. Three seeds per model, so the numbers have error bars.

    `tag` is what separates two runs that are otherwise identical. The mixture
    ablation is exactly that shape -- same model, same regime, same seed, a
    different training FILE -- and without a tag both arms write to run id
    "gemma4-e2b-full-s1" and the second replaces the first. "run" is the plain
    default and adds nothing to the name, so existing run ids are unaffected.
    """
    return _train(model=model, dataset=dataset, seed=seed, epochs=epochs,
                  tag=tag, lora_r=lora_r, lora_alpha=lora_alpha, lr=lr)


def _train(model: str, dataset: str, seed: int, epochs: int, tag: str,
           limit: int | None = None, max_steps: int = -1,
           lora_r: int = 16, lora_alpha: int = 0, lr: float = 2e-4) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    if model not in MODELS:
        raise KeyError(f"{model} adalah model pembanding, bukan model latih; "
                       f"pilih dari {sorted(MODELS)}")
    model_id = MODELS[model]
    train_file, test_file = DATASETS[dataset]
    if train_file is None:
        raise SystemExit(
            f"dataset '{dataset}' hanya untuk evaluasi, tidak punya sisi latih. "
            f"Melatih di atasnya berarti melatih model pada kunci jawaban "
            f"benchmark utamanya sendiri.")
    train_path = f"/data/{train_file}"

    set_seed(seed)
    started = time.time()
    # Any tag lands in the run id, not just "smoke". Two runs that differ only
    # in the dataset FILE -- same model, same regime, same seed, which is what
    # an ablation on training mixture looks like -- otherwise share a run_id,
    # and the second silently overwrites the first's adapter, manifest and
    # predictions. The ablation would then consist of one arm.
    run_id = f"{model}-{dataset}-s{seed}" + (
        f"-{tag}" if tag and tag != "run" else "")
    out_dir = f"/runs/{run_id}"

    # Hashed here, next to the read, not down in the manifest an hour later.
    # The volume is writable while a run is in flight -- upload() exists to
    # replace these very files -- so a hash taken after training describes
    # whatever is on the volume at the end, and would record the new dataset
    # as the provenance of a model trained on the old one. Nothing would look
    # wrong: the field is present, the hash is real, and it is the wrong hash.
    train_sha = sha256(train_path)
    rows = read_jsonl(train_path)
    if limit:
        rows = rows[:limit]
    rows, held = carve_validation(rows, seed)
    ds = Dataset.from_list(as_prompt_completion(rows))
    eval_ds = Dataset.from_list(as_prompt_completion(held))
    print(f"{len(ds)} contoh latih dari {train_file}, {len(held)} disisihkan "
          f"untuk validasi")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base, auto_class = load_base(model_id)
    base.config.use_cache = False

    names = linear_names(base)
    targets = select_targets(names)
    shape = check_targets(targets, names)
    print(f"LoRA pada {shape['modules']} modul di {shape['layers']} lapis "
          f"menara teks: {shape['shapes']}")

    # trl 1.12 takes a step count, not a fraction, so the fraction is applied
    # here against however many steps this particular run will take. A fixed
    # number would mean the 30-step smoke run spent all of it warming up.
    total_steps = (max_steps if max_steps > 0
                   else -(-len(ds) // (BATCH * ACCUM)) * epochs)
    # alpha follows rank unless asked otherwise. The scaling a LoRA applies is
    # alpha/r, so raising r while holding alpha fixed quietly halves the update
    # and reports itself as "rank did not help" -- a sweep result that is an
    # artefact of the constant, not of the rank.
    lora_alpha = lora_alpha or 2 * lora_r
    warmup = max(1, int(total_steps * WARMUP_FRACTION))

    trainer = SFTTrainer(
        model=base,
        train_dataset=ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules=targets,
        ),
        args=SFTConfig(
            output_dir=out_dir,
            num_train_epochs=epochs,
            max_steps=max_steps,
            per_device_train_batch_size=BATCH,
            gradient_accumulation_steps=ACCUM,
            gradient_checkpointing=True,
            # Without this the checkpointed blocks are re-run under no_grad and
            # the adapter never sees a gradient at all: the run completes, the
            # loss sits flat, and nothing reports an error.
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_steps=warmup,
            logging_steps=10,
            # Once per epoch, which is the granularity the question needs: is
            # three epochs too many? Cheap because it is forward passes on 5
            # percent of the data, not generation.
            eval_strategy="epoch",
            per_device_eval_batch_size=BATCH,
            # One adapter per epoch, which turns an epoch sweep from N runs
            # into one. Eval loss barely moved between epoch 2 (0.532) and
            # epoch 3 (0.496) while the third epoch costs a third of the run,
            # so "is epoch 3 worth paying for" is a budget question -- and it
            # cannot be answered by loss, only by scoring each checkpoint.
            # A LoRA adapter is a few tens of MB, so keeping all three is free
            # next to the GPU hour that produced them.
            save_strategy="epoch",
            save_total_limit=None,
            bf16=True,
            max_length=768,
            packing=False,
            # The point of the prompt/completion shape above. Explicit rather
            # than left to the default, because if this ever flips to False the
            # model trains on its own questions and nothing visibly breaks.
            completion_only_loss=True,
            report_to=[],
            seed=seed,
            data_seed=seed,
        ),
    )

    # Read from the wrapped model rather than recomputed from the target list:
    # this is what the optimiser will actually update, so a mismatch between
    # the two would be a real bug rather than a reporting detail.
    trainable, total = trainer.model.get_nb_trainable_parameters()

    result = trainer.train()
    trainer.model.save_pretrained(f"{out_dir}/adapter")
    tokenizer.save_pretrained(f"{out_dir}/adapter")

    history = [h for h in trainer.state.log_history if "loss" in h]
    # Separate key, so these do not silently mix into the training curve: an
    # eval entry carries "eval_loss", not "loss", and would be dropped by the
    # filter above without anyone noticing the validation data was wasted.
    eval_history = [h for h in trainer.state.log_history if "eval_loss" in h]
    elapsed = time.time() - started
    runtime = result.metrics.get("train_runtime", elapsed)
    steps = trainer.state.global_step or total_steps
    gpu = torch.cuda.get_device_name(0)
    manifest = {
        "run_id": run_id,
        "tag": tag,
        "model": model,
        "model_id": model_id,
        "auto_class": auto_class,
        "dataset": dataset,
        "train_file": train_file,
        "test_file": test_file,
        "train_sha256": train_sha,
        "examples": len(ds),
        "seed": seed,
        "epochs": epochs,
        "learning_rate": lr,
        "max_steps": max_steps,
        "total_steps": total_steps,
        "warmup_steps": warmup,
        "batch": BATCH,
        "grad_accum": ACCUM,
        # Dibaca dari argumen, bukan ditulis mati. Sebelumnya baris ini selalu
        # melaporkan r=16 alpha=32, sehingga manifest run r8 dan r32 menyebut
        # angka yang bukan miliknya sementara adapter_config.json di sebelahnya
        # menyebut yang benar. Manifest yang salah lebih buruk daripada tidak
        # ada, karena ia terbaca seperti catatan yang sudah diperiksa.
        "lora": {"r": lora_r, "alpha": lora_alpha, "dropout": 0.05, **shape},
        "trainable_params": trainable,
        "total_params": total,
        "trainable_share_pct": round(trainable / total * 100, 4),
        "first_loss": round(history[0]["loss"], 4) if history else None,
        "last_loss": round(history[-1]["loss"], 4) if history else None,
        "loss_history": history,
        "validation_examples": len(held),
        "eval_loss_history": eval_history,
        "best_epoch": (min(eval_history, key=lambda h: h["eval_loss"])["epoch"]
                       if eval_history else None),
        "train_runtime_s": round(runtime, 1),
        "total_s": round(elapsed, 1),
        "steps_done": steps,
        "seconds_per_step": round(runtime / steps, 3),
        "gpu": gpu,
        "usd_per_hour": rate_for(gpu),
        "estimated_cost_usd": round(elapsed / 3600 * rate_for(gpu), 3),
        "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "versions": versions(),
    }
    Path(f"{out_dir}/manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    runs.commit()

    print(json.dumps({k: v for k, v in manifest.items() if k != "loss_history"},
                     indent=2, ensure_ascii=False))
    return manifest


@app.function(image=image, gpu=GPU, volumes=VOLUMES, timeout=60 * 60 * 2)
def predict(model: str = "gemma4-e2b", run_id: str = "", dataset: str = "split70",
            limit: int = 0, max_new_tokens: int = 1024,
            checkpoint: str = "", metode: str = "generasi",
            prompt: str = "lagu") -> str:
    """Answer the test questions and write predictions the local scorer can read.

    Called with run_id empty this runs the untouched base model, which is the
    baseline every trained number is measured against. Same function either
    way, so the baseline and the result cannot drift apart in how they were
    produced.

    WHY THE TOKEN BUDGET IS GENEROUS

    It was 96, and at 96 the baseline was crippled by the measurement rather
    than by ignorance: an untrained instruct model answers "Lagu X umumnya
    dinyanyikan dengan nada dasar C Mayor atau G Mayor, tergantung..." and 56
    percent of its replies were still mid-sentence when the budget ran out. The
    fine-tuned model answers in three words and never reaches the limit. Any
    ceiling low enough to cut one side and not the other is not measuring
    knowledge, it is measuring verbosity, and it would have inflated every
    trained-versus-baseline gap in the thesis.

    Raising it to 256 fixed the trained side and looked fixed, because the
    trained side was the side being checked. On the 1002-item full test set the
    untouched base still lost 22.1 percent of its replies to the ceiling, so
    the default is now 1024. That costs the base nothing in accuracy it did not
    earn and costs the trained runs nothing at all, since they stop at EOS long
    before the limit and never pay for headroom they do not use.

    The truncation rate is reported per run so the next person can see whether
    the budget was still binding rather than take this comment's word for it.

    WHY THE ROWS ARE NOT REORDERED FOR BATCHING

    A batch runs until every sequence in it stops, so one rambling reply drags
    fifteen finished ones along at full price, and sorting by category to put
    the ramblers together looks like free money. It is not: generate.py already
    writes the test file in contiguous category blocks, so batches are grouped
    that way as they come and the sort is a no-op.

    Rows therefore go out in file order and must stay that way. The scorer
    walks predictions in file order and consumes the keys for a (song,
    category) slot one at a time -- 82 slots in the full test set hold more
    than one question -- so a file written in any other order pairs answers
    with a sibling's key, silently, with every line present and every id valid.
    evaluate.check_alignment is the alarm if this is ever changed.
    """
    import torch
    from importlib.metadata import version
    from peft import PeftModel
    from transformers import AutoTokenizer

    model_id = resolve(model)
    raised = budget_for(model, max_new_tokens)
    if raised != max_new_tokens:
        print(f"anggaran token dinaikkan {max_new_tokens} -> {raised} untuk {model}")
        max_new_tokens = raised
    _, test_file = DATASETS[dataset]
    # Printed because a prediction file is only as trustworthy as the questions
    # that produced it, and nothing else in the output records them. Two
    # workspaces hold their own copy of this volume and upload() can replace it
    # between runs, so "both accounts ran the same benchmark" is an assumption
    # until these two hashes are compared. Rows that came from different test
    # files still merge into one leaderboard without a single error.
    test_sha = sha256(f"/data/{test_file}")
    print(f"berkas uji {test_file} sha256 {test_sha}")
    rows = read_jsonl(f"/data/{test_file}")

    # An MC file stores questions and options, not chat turns, and its first
    # line is a header. Prompts are built with the same function the scorer
    # imports, so the two cannot drift apart.
    if prompt not in PROMPT_SISTEM:
        raise SystemExit(f"prompt {prompt!r} tidak dikenal, "
                         f"pilih {sorted(PROMPT_SISTEM)}")
    sistem = PROMPT_SISTEM[prompt]
    if dataset in MC_DATASETS:
        rows = [r for r in rows if "opsi" in r]
        for r in rows:
            r["messages"] = mc_messages(r, sistem) + [{"role": "assistant",
                                                       "content": r["kunci"]}]
        print(f"prompt sistem '{prompt}': {sistem}")
    elif prompt != "lagu":
        # The free-form track's prompts come out of the file, already written,
        # so there is no system message here to swap. Failing is better than
        # accepting the flag and producing a file whose name claims a condition
        # that was never applied.
        raise SystemExit(f"--prompt hanya berlaku untuk {sorted(MC_DATASETS)}")
    if limit:
        rows = rows[:limit]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base, _ = load_base(model_id)
    if run_id:
        # A named checkpoint scores an intermediate epoch. save_strategy is
        # "epoch", so one training run yields every epoch as its own adapter
        # and an epoch sweep costs generation instead of GPU hours.
        adapter = (f"/runs/{run_id}/{checkpoint}" if checkpoint
                   else f"/runs/{run_id}/adapter")
        if not Path(adapter).is_dir():
            ada = sorted(p.name for p in Path(f"/runs/{run_id}").glob("*"))
            raise SystemExit(f"tidak ada {adapter}. yang tersedia: {ada}")
        base = PeftModel.from_pretrained(base, adapter)
    base.eval()

    # The method goes in the name. A probability-scored file and a
    # generation-scored file answer the same questions by different means and
    # are not comparable, so letting one overwrite the other would silently mix
    # two measurements into one leaderboard column.
    suffix = f"-{checkpoint}" if checkpoint else ""
    if metode != "generasi":
        suffix += f"-{metode}"
    # The prompt condition goes in the name for the same reason the method
    # does. The two conditions ask the identical questions of the identical
    # weights and differ only in the system message, so the files are otherwise
    # indistinguishable and the second would overwrite the first -- leaving one
    # measurement on disk and a comparison that cannot be made.
    if prompt != "lagu":
        suffix += f"-{prompt}"
    name = f"{(run_id or model + '-base') + suffix}--{dataset}.jsonl"
    out_path = f"/runs/predictions/{name}"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    truncated = 0

    # The option-scoring cap, and why it is not one number.
    #
    # peluang-opsi truncates the prompt and the prompt-plus-option to the same
    # length. When a prompt alone reaches the cap, the option contributes no
    # tokens at all: every option scores an identical zero and max() hands the
    # answer to whichever letter sorts first. No error, no truncation count, and
    # the damage lands only on the longest questions -- which on IndoMMLU means
    # the reading-comprehension subjects and nothing else.
    #
    # LaguQA keeps 900 because every prediction file already on disk was made at
    # 900 and a wider cap would produce numbers that no longer match them. The
    # external sets are new, have no such history, and reach 3.400 characters at
    # their longest, so they get room. tanpa_opsi below is the alarm either way.
    cap = 900 if dataset == "mc" else 2048
    tanpa_opsi = 0
    #
    # WHY MULTIPLE CHOICE IS SCORED BY PROBABILITY, NOT BY GENERATED TEXT
    #
    # Reading the answer out of generated text measures how a model likes to
    # write, not what it knows. Two of the comparison models reason in the open
    # without <think> tags -- "The user is asking about the highest note..." --
    # and both ran past a 1024-token budget still mid-thought, never reaching an
    # answer. Scored on that text they would have recorded near zero and been
    # written up as knowing nothing about Indonesian songs, when what was
    # measured was verbosity. Raising the budget only moves the threshold and
    # multiplies the bill: those runs were heading for three hours each.
    #
    # The answer to a five-option question is one letter, so the model's
    # probability for each letter at the first answer position is the whole
    # measurement. One forward pass, nothing generated, nothing to truncate,
    # and no model is punished for showing its working. This is how MMLU-style
    # benchmarks have always scored multiple choice.
    #
    # Every model on the leaderboard must be scored the same way for the column
    # to mean anything, so the free-form track keeps generation and the
    # multiple-choice track uses this.
    huruf_id: dict[str, list[int]] = {}
    if metode == "peluang":
        for h in "ABCDE":
            ids = set()
            # Both bare and space-prefixed: tokenisers differ on which one
            # follows a chat template, and picking the wrong one would compare
            # a real logit against whatever id 0 happens to be.
            for varian in (h, f" {h}"):
                t = tokenizer.encode(varian, add_special_tokens=False)
                if t:
                    ids.add(t[0])
            if not ids:
                raise SystemExit(f"tokenizer tidak punya token untuk huruf {h}")
            huruf_id[h] = sorted(ids)
        print(f"metode peluang, token per huruf: "
              f"{ {h: len(v) for h, v in huruf_id.items()} }")

    batch = 16
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        prompts = [
            tokenizer.apply_chat_template(r["messages"][:-1], tokenize=False,
                                          add_generation_prompt=True)
            for r in chunk
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=768).to(base.device)

        if metode == "peluang-opsi":
            # Score the option TEXT, not the letter. A model that has never
            # been taught the A/B/C/D/E convention can still know which answer
            # is right, and letter scoring would hide that behind its letter
            # bias. For each option the mean log-probability of its tokens
            # given the prompt is computed, and the highest wins. Mean rather
            # than sum, so a long option is not penalised for being long.
            # All five options of one item go through together. They share the
            # prompt, so with right padding the prompt occupies the same
            # positions in every row and one offset serves all five. Scoring
            # them one at a time worked but cost five forward passes per
            # question, which put a single model at half an hour.
            sisi = tokenizer.padding_side
            tokenizer.padding_side = "right"
            for r in chunk:
                opsi = r["opsi"]
                huruf = sorted(opsi)
                dasar = tokenizer.apply_chat_template(
                    r["messages"][:-1], tokenize=False, add_generation_prompt=True)
                n_awal = len(tokenizer(dasar, truncation=True,
                                       max_length=cap)["input_ids"])
                enc5 = tokenizer([dasar + str(opsi[h]) for h in huruf],
                                 return_tensors="pt", padding=True,
                                 truncation=True, max_length=cap).to(base.device)
                # Only the option's own positions are ever read, so only those
                # are asked for. Materialising every position costs
                # 5 x seq x vocab in bf16, which at a 262k vocabulary and a
                # 2048-token cap is 5.4 GB of logits thrown away one line
                # later; on a 24 GB L4 the allocator spent the IndoMMLU run
                # thrashing and warning about it. Keeping the tail makes it
                # 5 x (option + 1) x vocab, tens of megabytes.
                #
                # The slice is the same one as before. logits_to_keep=k returns
                # positions seq-k .. seq-1, and k here is chosen so that is
                # exactly n_awal-1 .. seq-1, which [:, :-1] then trims to the
                # old n_awal-1 .. seq-2. Clamped at 1 because 0 means "keep
                # everything" and a prompt that filled the cap would otherwise
                # ask for all of them again; tanpa_opsi catches that item.
                sisa = max(1, enc5["input_ids"].shape[1] - n_awal + 1)
                with torch.no_grad():
                    try:
                        lg = base(**enc5, logits_to_keep=sisa).logits
                    except TypeError:
                        try:
                            lg = base(**enc5, num_logits_to_keep=sisa).logits
                        except TypeError:
                            lg = base(**enc5).logits[:, n_awal - 1:, :]
                lp = torch.log_softmax(lg[:, :-1, :], dim=-1)
                ids = enc5["input_ids"][:, n_awal:]
                mask = enc5["attention_mask"][:, n_awal:]
                if int(mask.sum().item()) == 0:
                    tanpa_opsi += 1
                ambil = lp.gather(2, ids.unsqueeze(-1)).squeeze(-1) * mask
                # Mean over the option's own tokens, so a long option is not
                # penalised for being long.
                rerata = ambil.sum(1) / mask.sum(1).clamp(min=1)
                skor = {h: rerata[k].item() for k, h in enumerate(huruf)}
                pilih = max(skor, key=lambda h: skor[h])
                keluar = {"id_lagu": r["id_lagu"], "kategori": r["kategori"],
                          "prediksi": pilih}
                if "id" in r:
                    keluar = {"id": r["id"], **keluar}
                lines.append(json.dumps(keluar, ensure_ascii=False))
            tokenizer.padding_side = sisi
            if i % (batch * 10) == 0:
                print(f"{i + len(chunk)}/{len(rows)}")
            continue

        if metode == "peluang":
            # One forward pass, no generation: read off which option letter the
            # model puts the most probability on. Padding is left-aligned, so
            # the last position is the real end of every prompt.
            #
            # logits_to_keep=1 is not an optimisation, it is what makes this
            # run at all. Asking for every position materialises a tensor of
            # [16 x 768 x vocab]; at a 256k vocabulary in bf16 that is 6.3 GB
            # of logits thrown away one line later, and the 9B models died of
            # it on a 24 GB card. Keeping one position makes it 8 MB. Older
            # signatures call the argument num_logits_to_keep, and a model that
            # accepts neither falls back to a smaller batch rather than to a
            # crash.
            with torch.no_grad():
                try:
                    logits = base(**enc, logits_to_keep=1).logits[:, -1, :]
                except TypeError:
                    try:
                        logits = base(**enc, num_logits_to_keep=1).logits[:, -1, :]
                    except TypeError:
                        logits = base(**enc).logits[:, -1, :]
            for r, baris_logit in zip(chunk, logits):
                skor = {h: max(baris_logit[t].item() for t in ids)
                        for h, ids in huruf_id.items()}
                pilih = max(skor, key=lambda h: skor[h])
                keluar = {"id_lagu": r["id_lagu"], "kategori": r["kategori"],
                          "prediksi": pilih}
                if "id" in r:
                    keluar = {"id": r["id"], **keluar}
                lines.append(json.dumps(keluar, ensure_ascii=False))
            if i % (batch * 10) == 0:
                print(f"{i + len(chunk)}/{len(rows)}")
            continue

        with torch.no_grad():
            # Greedy on purpose: the scorer compares against one key, so
            # sampling would add variance that has nothing to do with the model.
            got = base.generate(**enc, max_new_tokens=max_new_tokens,
                                do_sample=False,
                                pad_token_id=tokenizer.pad_token_id)
        for r, seq in zip(chunk, got):
            new = seq[enc["input_ids"].shape[1]:]
            text = tokenizer.decode(new, skip_special_tokens=True).strip()
            # A reply that used every token it was given probably had more to
            # say. Counted rather than assumed, because this is the one number
            # that says whether the budget above is still doing damage.
            if int((new != tokenizer.pad_token_id).sum()) >= max_new_tokens:
                truncated += 1
            # These three field names stay Indonesian while everything else in
            # this file is English. They are not new: they are the schema of
            # the frozen benchmark, which evaluate.py reads and whose SHA-256
            # is recorded in laguqa_mc_manifest.json. Renaming them for tidiness
            # would break the scorer and invalidate a published hash.
            keluar = {"id_lagu": r["id_lagu"], "kategori": r["kategori"],
                      "prediksi": text}
            # MC items carry their own id because one song-category slot holds
            # many of them; the scorer pairs on that id, not on the slot.
            if "id" in r:
                keluar = {"id": r["id"], **keluar}
            lines.append(json.dumps(keluar, ensure_ascii=False))
        if i % (batch * 10) == 0:
            print(f"{i + len(chunk)}/{len(rows)}")

    # Every prediction file carries the hash of the questions it answered.
    # Slots survive a rebuild -- same songs, same categories, same count -- so
    # a scorer pairing on slot alone will score last week's answers against
    # this week's questions and report an ordinary-looking number. A stale
    # qwen file was caught only because its sampled songs happened to differ,
    # which is luck rather than a check.
    #
    # The decoding settings ride along in the same header. A leaderboard row is
    # only reproducible if the settings that produced it are known, and putting
    # them in the prediction file means they cannot drift apart from the
    # answers the way a separately-kept note would. `berpikir` records that no
    # enable_thinking flag was forced: each model keeps whatever its own chat
    # template defaults to, which is the only setting that is fair across a
    # leaderboard mixing reasoning and non-reasoning models.
    lines.insert(0, json.dumps({
        "berkas_uji": test_file, "sha256": test_sha,
        "model": model, "model_id": model_id,
        # Which adapter, not just which base. Without it a file that has been
        # renamed cannot say whether it came from final or lr4e4, and the two
        # differ by four points.
        "run_id": run_id or "",
        "checkpoint": checkpoint or "",
        "metode": metode,
        # Recorded, not only encoded in the file name. A file that is renamed
        # or moved still says which system message produced it, and the two
        # forgetting conditions are otherwise identical in every field here.
        # Null outside the multiple-choice track: there the system message
        # comes out of the test file itself, and naming one here would record a
        # prompt that this run never applied.
        "prompt": prompt if dataset in MC_DATASETS else None,
        "prompt_sistem": sistem if dataset in MC_DATASETS else None,
        "batas_token_prompt": cap if metode == "peluang-opsi" else None,
        "dekode": ("argmax logit huruf, tanpa generasi" if metode == "peluang"
                   else "greedy (do_sample=False)"),
        "suhu": None, "top_p": None, "top_k": None, "num_beams": 1,
        "presisi": "bfloat16", "kuantisasi": "tidak ada",
        "maks_token_baru": max_new_tokens,
        "berpikir": "bawaan templat masing-masing model",
        "batch": batch,
        "transformers": version("transformers"),
        "torch": torch.__version__,
    }, ensure_ascii=False))
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    runs.commit()
    share = truncated / len(lines) * 100 if lines else 0.0
    print(f"ditulis {len(lines)} prediksi ke {out_path}")
    print(f"terpotong di batas {max_new_tokens} token: {truncated} ({share:.1f}%)")
    if share > 10:
        print("PERINGATAN: anggaran token masih mengikat. jawaban yang terpotong "
              "dinilai atas apa yang sempat keluar, bukan atas apa yang diketahui.")
    if tanpa_opsi:
        # Not a warning that can be weighed against the result: these items were
        # decided by alphabetical order, not by the model.
        print(f"GAGAL: {tanpa_opsi} soal kehabisan ruang di batas {cap} token "
              f"sehingga teks opsinya tidak ikut dinilai. Naikkan batasnya dan "
              f"ulangi; berkas ini tidak sah.")
    return out_path


# --- local entrypoints -------------------------------------------------------


@app.local_entrypoint()
def upload():
    """Push the benchmark files to the data volume, and record their hashes.

    On a volume rather than baked into the image so the dataset can be replaced
    without rebuilding, and so the hash printed here is the one every manifest
    will carry.
    """
    files = sorted({f for f in sum(DATASETS.values(), ()) if f})
    with data_vol.batch_upload(force=True) as batch:
        for name in files:
            local = local_dataset(name)
            batch.put_file(local, f"/{name}")
            print(f"{name:34} {local.stat().st_size / 1e6:7.2f} MB  {sha256(local)[:16]}")
    print(f"\n{len(files)} berkas di volume laguqa-data")


@app.local_entrypoint()
def experiment(models: str = "gemma4-e2b,sealion-e2b,gemma4-e4b",
               seeds: str = "1,2,3", dataset: str = "split70",
               epochs: int = 3, tag: str = "run",
               predict_after: bool = True):
    """Run the whole grid in parallel and score nothing -- that happens locally.

    Parallel because the cost is per GPU-second either way, so nine runs at
    once cost what nine runs in sequence cost and finish in a ninth of the
    wall time. The burn RATE is nine times higher, which is why the pilot runs
    first and alone: a recipe error found on run one costs a dollar, and the
    same error found across a full parallel grid costs all of them.

    Predictions follow each run in the same call, so a finished adapter that
    was never asked a question cannot sit unnoticed in the volume.
    """
    grid = [(m.strip(), int(s))
            for m in models.split(",") if m.strip()
            for s in seeds.split(",") if s.strip()]
    unknown = sorted({m for m, _ in grid} - set(MODELS))
    if unknown:
        raise SystemExit(f"bukan model latih: {unknown}. pilih dari {sorted(MODELS)}")

    print(f"{len(grid)} percobaan: {grid}\n")
    # tag reaches the run id, so a grid rerun on different training data does
    # not overwrite the first grid one adapter at a time.
    handles = [run.spawn(model=m, dataset=dataset, seed=s, epochs=epochs,
                         tag=tag)
               for m, s in grid]

    done, failed = [], []
    for (m, s), handle in zip(grid, handles):
        try:
            got = handle.get()
        except Exception as exc:  # noqa: BLE001 - one failure must not sink the rest
            failed.append((m, s, f"{type(exc).__name__}: {exc}"))
            print(f"[GAGAL] {m} benih {s}: {exc}")
            continue
        done.append(got)
        print(f"[selesai] {got['run_id']:34} loss {got['first_loss']} -> "
              f"{got['last_loss']}  {got['train_runtime_s']}s  "
              f"${got['estimated_cost_usd']}")

    if predict_after and done:
        print(f"\nprediksi untuk {len(done)} adapter")
        jobs = [predict.spawn(model=g["model"], run_id=g["run_id"],
                              dataset=dataset) for g in done]
        for job in jobs:
            print(f"  {job.get()}")

    total = sum(g["estimated_cost_usd"] for g in done)
    print(f"\n{len(done)} berhasil, {len(failed)} gagal, "
          f"perkiraan biaya latih ${total:.2f}")
    for m, s, why in failed:
        print(f"  gagal: {m} benih {s} -- {why}")
    print("ambil hasilnya dengan modal run modal_train.py::fetch")


@app.local_entrypoint()
def forgetting(adapters: str = "", model: str = "gemma4-e2b",
               datasets: str = "indommlu,indoculture",
               prompts: str = "lagu,netral"):
    """Every cell of the forgetting grid, in one call.

    The base model is always included and is not optional. An adapter's score
    on IndoMMLU means nothing on its own -- what is being measured is the
    change from the untouched model, so a grid without the base arm produces
    numbers that cannot be read.

    Both prompt conditions are run for the same reason. The adapter has only
    ever seen questions introduced by the LaguQA system message, so a drop
    under that message alone could be lost knowledge or could be the label, and
    only the pair separates them.

    Parallel, because the bill is per GPU-second and these are minutes each.

    The benchmarking account caps GPU containers at 10, so a grid larger than
    that runs in waves. Modal queues the surplus rather than refusing it, and
    the bill does not change because it is per GPU-second either way. What does
    change is wall time, and anything launched while a large grid is in flight
    waits behind it.

    `model` takes a list, so the same call also sweeps the comparison models
    with no adapter attached. Those arms answer a different question from the
    forgetting one: whether a model's score on an existing Indonesian benchmark
    predicts its score here. Adapters are only meaningful against their own
    base, so an adapter list and a multi-model list are not combined.

        modal run modal_train.py::forgetting --adapters gemma4-e2b-full-s1-final
        modal run modal_train.py::forgetting --model qwen35-9b,granite42-8b \
            --prompts netral
    """
    pakai = [r.strip() for r in adapters.split(",") if r.strip()]
    models = [m.strip() for m in model.split(",") if m.strip()]
    if pakai and len(models) > 1:
        raise SystemExit("adapter hanya berlaku untuk satu model dasar; "
                         "jalankan sapuan pembanding tanpa --adapters")
    ds = [d.strip() for d in datasets.split(",") if d.strip()]
    pr = [p.strip() for p in prompts.split(",") if p.strip()]
    salah = sorted(set(ds) - MC_DATASETS)
    if salah:
        raise SystemExit(f"bukan berkas pilihan ganda: {salah}")
    salah = sorted(set(pr) - set(PROMPT_SISTEM))
    if salah:
        raise SystemExit(f"prompt tidak dikenal: {salah}")

    grid = [(m, r, d, p) for m in models for r in [""] + pakai
            for d in ds for p in pr]
    print(f"{len(grid)} prediksi: {len(models)} model x {len(pakai) + 1} "
          f"varian x {len(ds)} benchmark x {len(pr)} prompt\n")
    jobs = [(sel, predict.spawn(model=sel[0], run_id=sel[1], dataset=sel[2],
                                metode="peluang-opsi", prompt=sel[3]))
            for sel in grid]

    gagal = []
    for (m, r, d, p), job in jobs:
        nama = r or f"{m}-base"
        try:
            print(f"[selesai] {nama:34} {d:12} {p:7} -> {job.get()}")
        except Exception as exc:  # noqa: BLE001 - one cell must not sink the rest
            gagal.append((nama, d, p, f"{type(exc).__name__}: {exc}"))
            print(f"[GAGAL]   {nama:34} {d:12} {p:7} -- {exc}")
    for nama, d, p, why in gagal:
        print(f"  gagal: {nama} {d} {p} -- {why}")
    print(f"\n{len(jobs) - len(gagal)} berhasil, {len(gagal)} gagal")
    print("ambil hasilnya dengan modal run modal_train.py::fetch, "
          "lalu python scripts/27_external_report.py")


@app.local_entrypoint()
def bench(model: str = "gemma4-e2b", gpus: str = "L4,A100-40GB,L40S"):
    """Time the same 30 steps on several cards and cost out the full experiment.

    A faster card is not automatically a more expensive one. An L4 is the
    cheapest per hour and could still be the most expensive per run if it takes
    three times as long. Nine training runs make that difference worth thirty
    cents to measure instead of guess.
    """
    steps_full = -(-10000 // (BATCH * ACCUM)) * 3  # split70, three epochs
    rows = []
    for gpu in [g.strip() for g in gpus.split(",") if g.strip()]:
        got = smoke.with_options(gpu=gpu).remote(model=model)
        per_step = got["seconds_per_step"]
        hours = per_step * steps_full / 3600
        rows.append((gpu, got["gpu"], per_step, got["peak_memory_gb"],
                     hours, hours * got["usd_per_hour"]))

    print(f"\n{'diminta':12} {'didapat':26} {'detik/langkah':>13} "
          f"{'memori':>7} {'jam penuh':>10} {'usd/run':>8}")
    for asked, got_name, per_step, mem, hours, cost in rows:
        print(f"{asked:12} {got_name:26} {per_step:>13.2f} {mem:>6.1f}G "
              f"{hours:>10.2f} {cost:>8.2f}")
    if rows:
        best = min(rows, key=lambda r: r[5])
        print(f"\ntermurah per run: {best[0]} pada ${best[5]:.2f}, "
              f"sembilan run ${best[5] * 9:.2f}")


CARD = """---
license: {license}
base_model: {model_id}
library_name: peft
tags:
  - lora
  - laguqa
  - indonesian
  - music
language:
  - id
---

# {run_id}

Adapter LoRA untuk `{model_id}`, dilatih pada LaguQA: 107 lagu daerah dan wajib
nasional Indonesia yang ditranskripsi dari satu buku cetak ke notasi ABC 2.1.

Ini adapter, bukan model utuh. Bobot dasarnya tetap milik penerbit model dan
harus diunduh terpisah.

## Cara memakai

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{model_id}", dtype="bfloat16")
model = PeftModel.from_pretrained(base, "{repo}")
tok = AutoTokenizer.from_pretrained("{repo}")
```

## Data latih

| | |
|---|---|
| Berkas | `{train_file}` |
| SHA-256 | `{train_sha}` |
| Contoh | {examples} |
| Regime | `{dataset}` |

{regime_note}

## Latihan

| | |
|---|---|
| LoRA r / alpha / dropout | {r} / {alpha} / {dropout} |
| Modul disasar | {modules} di {layers} lapis menara teks |
| Parameter dilatih | {trainable} |
| Batch x akumulasi | {batch} x {accum} |
| Epoch | {epochs} |
| Langkah | {steps} ({warmup} pemanasan) |
| Learning rate | 2e-4, cosine |
| Benih | {seed} |
| Kartu | {gpu} |
| Waktu | {runtime} detik |
| Loss awal -> akhir | {first_loss} -> {last_loss} |

Kerugian hanya dihitung pada giliran asisten (`completion_only_loss=True`),
sehingga yang dihafal adalah jawabannya, bukan pertanyaannya.

## Hasil

{results}

## Batasan yang harus dibaca sebelum memakai

Diturunkan dari `SOURCE.md` pada repositori dataset dan ikut terbawa ke setiap
angka di atas.

- **Delapan belas kata di lirik masih tergabung salah**, misalnya bentuk seperti
  `IndoNesia`, karena satu suku kata tertranskripsi berhuruf kapital. Kunci
  jawaban lirik untuk kata-kata itu memuat ejaan yang keliru, jadi model yang
  menjawab benar justru dinilai meleset di soal tersebut.
- **Birama 50 lagu disimpulkan, bukan dibaca**, karena halaman bergaya `1 = C`
  tidak mencetak tanda biramanya. Tidak ada soal birama yang dibangkitkan dari
  lagu-lagu itu, tetapi kolomnya tetap tidak boleh dipakai sebagai fakta.
- **Dua puluh delapan berkas ABC belum terverifikasi**, sehingga soal penalaran
  notasi hanya menjangkau 79 lagu.
- **Pencipta dan asal daerah masing-masing hanya terisi 55 dari 107 baris.**
  Buku mencantumkan pencipta untuk lagu nasional dan asal daerah untuk lagu
  daerah, hampir tidak pernah keduanya. Kekosongan itu jawaban sah "buku tidak
  mencantumkan", bukan data hilang.

Buku sumbernya terbit 2025 dan masih berhak cipta penuh. Yang diterbitkan hanya
metadata, transkripsi untuk keperluan analisis, dan potongan pendek di dalam
soal. Halaman pindaiannya tidak disertakan.

## Sitasi

```bibtex
@misc{{laguqa,
  author = {{Hendianto, Mohammad Farid}},
  title  = {{LaguQA: benchmark pengetahuan lagu Indonesia untuk model bahasa}},
  year   = {{2026}},
  note   = {{Skripsi, Program Studi Informatika, Universitas Ahmad Dahlan}}
}}
```

Sertakan juga buku sumbernya, lihat `SOURCE.md`.
"""

REGIME_NOTE = {
    "split70": "Dilatih pada 70 lagu, diuji pada 37 lagu yang tidak pernah "
               "dilihat. Angka di bawah mengukur pengetahuan yang benar-benar "
               "dipindahkan, bukan hafalan yang diputar ulang.",
    "full": "**Dilatih pada seluruh 107 lagu.** Tidak ada lagu yang ditahan, "
            "sehingga angka uji apa pun untuk varian ini tidak mengukur "
            "generalisasi dan tidak boleh dilaporkan seolah-olah begitu. "
            "Varian ini untuk dipakai, bukan untuk diukur.",
}


@app.local_entrypoint()
def release(run_id: str, dest: str = "rilis", license: str = "gemma",
            repo: str = ""):
    """Assemble one run into a directory ready to push to the Hub.

    Deliberately stops at building the directory. Pushing is a separate,
    irreversible act that publishes the author's name and an institution's
    under a licence, and it should be a human's decision made after reading
    what is in the folder, not a side effect of a training script finishing.

    The card is filled from the run's own manifest rather than typed, so the
    hyperparameters it advertises are the ones that ran. A card written by hand
    drifts from the run the first time a default changes.
    """
    out = Path(dest) / run_id
    (out).mkdir(parents=True, exist_ok=True)

    # Volume-relative, with no "/runs" in front. Inside a container the volume
    # is mounted at /runs, but these calls address the volume itself, and the
    # mount path is not part of the key.
    manifest = json.loads(b"".join(runs.read_file(f"{run_id}/manifest.json")))

    n = 0
    for entry in runs.iterdir(f"{run_id}/adapter"):
        target = out / Path(entry.path).name
        target.write_bytes(b"".join(runs.read_file(entry.path)))
        print(f"  {target.name:34} {target.stat().st_size / 1e6:8.2f} MB")
        n += 1
    if not n:
        raise SystemExit(f"tidak ada bobot adapter di /runs/{run_id}/adapter")

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Must match evaluate.write_markdown, which keeps the regime in the name so
    # a split37 score and a full-set score for one checkpoint cannot overwrite
    # each other. A card that silently quotes the wrong regime is worse than a
    # card with no numbers, so a missing file is left to the check below.
    scores = Path("hasil") / f"{run_id}--{manifest['dataset']}--skor.md"
    lora = manifest["lora"]
    card = CARD.format(
        license=license, model_id=manifest["model_id"], run_id=run_id,
        repo=repo or f"<pengguna>/{run_id}",
        train_file=manifest["train_file"], train_sha=manifest["train_sha256"],
        examples=manifest["examples"], dataset=manifest["dataset"],
        regime_note=REGIME_NOTE.get(manifest["dataset"], ""),
        r=lora["r"], alpha=lora["alpha"], dropout=lora["dropout"],
        modules=lora["modules"], layers=lora["layers"],
        trainable=(f"{manifest['trainable_params']:,} dari "
                   f"{manifest['total_params']:,} "
                   f"({manifest['trainable_share_pct']}%)"
                   if "trainable_params" in manifest else "lihat manifest.json"),
        batch=manifest["batch"], accum=manifest["grad_accum"],
        epochs=manifest["epochs"], steps=manifest["steps_done"],
        warmup=manifest["warmup_steps"], seed=manifest["seed"],
        gpu=manifest["gpu"], runtime=manifest["train_runtime_s"],
        first_loss=manifest["first_loss"], last_loss=manifest["last_loss"],
        results=(scores.read_text(encoding="utf-8") if scores.exists()
                 else "_Belum dinilai. Jalankan predict lalu 11_evaluate.py, "
                      "lalu bangun ulang folder ini._"),
    )
    (out / "README.md").write_text(card, encoding="utf-8")

    print(f"\n{n + 2} berkas di {out}/")
    print("periksa README.md dulu, terutama bagian Batasan, sebelum diunggah.")


@app.local_entrypoint()
def fetch(dest: str = "hasil"):
    """Copy manifests and predictions back down, so scoring happens locally.

    Scoring runs against benchmark/evaluate.py on this machine, with the tests
    that pin its behaviour. Running it inside a training container would make
    the numbers depend on an image nobody kept.
    """
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)

    # Anything already filed under hasil/arsip/ was moved there on purpose:
    # it was measured on a benchmark version that no longer applies, and the
    # figures must not draw it. The volume still holds it, because deleting
    # evidence to make a glob behave is the wrong trade. So fetch leaves those
    # alone instead -- otherwise every fetch quietly undoes the archiving,
    # which is exactly what happened once.
    #
    # Matched by CONTENT, not by name. Matching by name broke both ways:
    # archived files were renamed with a hash suffix to keep two different
    # files of the same name apart, after which the name no longer matched and
    # every fetch restored the stale copy; and before that, a name match
    # blocked a legitimate re-run that happened to reuse the name. A hash
    # cannot do either. It only skips bytes that are already on disk, and a
    # genuinely new prediction has new bytes.
    arsip = {hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (out / "arsip").glob("*") if p.is_file()}
    skipped = 0

    def bring(target: Path, blob: bytes) -> bool:
        nonlocal skipped
        if hashlib.sha256(blob).hexdigest() in arsip:
            skipped += 1
            return False
        target.write_bytes(blob)
        print(target)
        return True

    n = 0
    for entry in runs.iterdir("/"):
        try:
            blob = b"".join(runs.read_file(f"{entry.path}/manifest.json"))
        except Exception:  # noqa: BLE001 - not every directory holds a run
            continue
        n += bring(out / f"{Path(entry.path).name}-manifest.json", blob)
    try:
        for entry in runs.iterdir("/predictions"):
            n += bring(out / Path(entry.path).name,
                       b"".join(runs.read_file(entry.path)))
    except Exception:  # noqa: BLE001 - no predictions yet
        pass
    print(f"\n{n} berkas ke {out}/"
          + (f", {skipped} dilewati karena sudah diarsipkan" if skipped else ""))
