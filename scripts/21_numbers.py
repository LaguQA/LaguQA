#!/usr/bin/env python3
"""Kumpulkan setiap angka yang boleh dikutip skripsi, langsung dari berkas hasil."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from laguqa.report.numbers import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
