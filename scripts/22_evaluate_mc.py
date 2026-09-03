#!/usr/bin/env python3
"""Nilai jawaban pilihan ganda LaguQA-MC terhadap kuncinya."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from laguqa.benchmark.evaluate_mc import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
