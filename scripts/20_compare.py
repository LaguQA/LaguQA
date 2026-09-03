#!/usr/bin/env python3
"""Bandingkan dua run atau lebih, satu baris per kategori, beserta selisihnya."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from laguqa.report.compare import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
