#!/usr/bin/env python3
"""Grafik analisis Bab IV dari tabel hasil dan manifes percobaan."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from laguqa.report.grafik import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
