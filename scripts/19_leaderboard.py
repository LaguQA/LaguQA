#!/usr/bin/env python3
"""Bangun papan skor LaguQA dari berkas prediksi di hasil/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from laguqa.report.leaderboard import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
