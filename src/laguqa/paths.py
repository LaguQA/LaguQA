"""Every path the package uses, resolved in one place.

Two roots, because the material splits in two along the licence line.

REPO_ROOT holds what can be published: the ABC transcriptions, the dataset
table, the frozen split, the generated benchmark. It is derived from this
file's location, so a checkout works wherever it sits.

WORKSPACE holds what cannot be published: the scanned pages of a copyrighted
songbook and the author's own manuscript files. Those live outside the
repository, in the directory containing it.

Every root can be redirected with an environment variable, so neither the
private layout nor an installed-package layout is forced on anyone:

    LAGUQA_DATA           the repository's own data/ directory
    LAGUQA_WORKSPACE      directory holding the private material
    LAGUQA_RAW_SCANS      photographed pages, untouched
    LAGUQA_SCANS          pages after flat-field correction; these are what
                          gets sent to the vision model
    LAGUQA_ABC_RAW        raw model output, one .abc per song, before the
                          validator has passed it
    LAGUQA_XLSX           the spreadsheet the dataset is rebuilt from
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    """Read a path from the environment, falling back to the default."""
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


# .../laguqa/src/laguqa/paths.py -> .../laguqa
REPO_ROOT = Path(__file__).resolve().parents[2]

# --- published material, inside the repository ------------------------------

DATA_DIR = _env_path("LAGUQA_DATA", REPO_ROOT / "data")
ABC_DIR = DATA_DIR / "abc"
CSV_PATH = DATA_DIR / "laguqa.csv"
SPLIT_PATH = DATA_DIR / "split.json"

BENCHMARK_DIR = DATA_DIR / "benchmark"
TRAIN_PATH = BENCHMARK_DIR / "laguqa_train.jsonl"
TEST_PATH = BENCHMARK_DIR / "laguqa_test.jsonl"

PROMPTS_DIR = REPO_ROOT / "prompts"
PROMPT_PATH = PROMPTS_DIR / "prompt.txt"
DOCS_DIR = REPO_ROOT / "docs"

# --- private material, outside the repository -------------------------------

WORKSPACE = _env_path("LAGUQA_WORKSPACE", REPO_ROOT.parent)

RAW_SCANS_DIR = _env_path("LAGUQA_RAW_SCANS", WORKSPACE / "sumber" / "halaman-mentah")
SCANS_DIR = _env_path("LAGUQA_SCANS", WORKSPACE / "sumber" / "halaman-siap")
ABC_RAW_DIR = _env_path("LAGUQA_ABC_RAW", WORKSPACE / "abc" / "gemini")
XLSX_PATH = _env_path("LAGUQA_XLSX", WORKSPACE / "naskah" / "dataset.xlsx")
