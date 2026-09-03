#!/usr/bin/env python3
"""Stage 28 - kemas rilis dataset menjadi zip untuk Kaggle.

Dibangun dari rilis-dataset/, bukan dari data/, supaya cermin Kaggle dan
repositori HuggingFace berisi berkas yang sama persis.

Dry run by default; pass --apply to write.

    python scripts/28_kaggle_release.py
    python scripts/28_kaggle_release.py --apply
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from laguqa.report.kaggle_release import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
