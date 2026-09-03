#!/usr/bin/env python3
"""Every knob on run() must actually reach _train().

run() is the entrypoint an experiment is launched with and _train() is where
the value is used, so a parameter added to one and not forwarded to the other
is accepted on the command line, ignored during the run, and recorded in the
manifest as whatever the default was. A sweep would then compare arms that were
never actually different, and nothing would look wrong.

That failure is invisible locally. It surfaced on Modal as an UnboundLocalError
several GPU minutes into a paid run, after the checkpoint had downloaded and
LoRA had attached. This test reads the file with ast, needs no GPU, and catches
it first. Modal wraps the decorated functions, so the source is parsed rather
than introspected through the wrapper.

Usage:
    python tests/test_modal_wiring.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

BERKAS = Path(__file__).resolve().parents[1] / "modal_train.py"

# Parameters of run() that _train() is not expected to receive.
BUKAN_TUNABLE: set[str] = set()

# Settings the manifest must record, because a result that cannot be traced
# back to the settings that produced it is not reproducible.
WAJIB_DI_MANIFES = ("learning_rate", "lora", "epochs", "seed", "train_sha256")


def fungsi(pohon: ast.Module, nama: str) -> ast.FunctionDef:
    for node in ast.walk(pohon):
        if isinstance(node, ast.FunctionDef) and node.name == nama:
            return node
    raise SystemExit(f"fungsi {nama} tidak ditemukan di {BERKAS.name}")


def params(node: ast.FunctionDef) -> list[str]:
    a = node.args
    return [p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)]


def main() -> int:
    pohon = ast.parse(BERKAS.read_text(encoding="utf-8"))
    run = fungsi(pohon, "run")
    train = fungsi(pohon, "_train")
    train_params = set(params(train))

    # The keyword names run() passes to _train().
    diteruskan: dict[str, str] = {}
    for node in ast.walk(run):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_train"):
            for kw in node.keywords:
                if kw.arg and isinstance(kw.value, ast.Name):
                    diteruskan[kw.arg] = kw.value.id

    bad = 0
    periksa = [p for p in params(run) if p not in BUKAN_TUNABLE]
    for nama in periksa:
        if nama not in train_params:
            print(f"[GAGAL] run() menerima '{nama}', _train() tidak punya "
                  f"parameter itu")
            bad += 1
        elif diteruskan.get(nama) != nama:
            print(f"[GAGAL] run() menerima '{nama}' tetapi tidak "
                  f"meneruskannya ke _train()")
            bad += 1
    print(f"{len(periksa)} parameter run() diperiksa, {bad} gagal")

    isi = ast.get_source_segment(BERKAS.read_text(encoding="utf-8"), train) or ""
    hilang = [m for m in WAJIB_DI_MANIFES if f'"{m}"' not in isi]
    for m in hilang:
        print(f"[GAGAL] manifes tidak mencatat '{m}'")
    bad += len(hilang)
    print(f"{len(WAJIB_DI_MANIFES)} medan manifes diperiksa, "
          f"{len(hilang)} hilang")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
