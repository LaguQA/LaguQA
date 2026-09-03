#!/usr/bin/env python3
"""Bakukan ejaan nama pencipta di data/laguqa.csv.

Lihat laguqa.dataset.composers untuk alasannya.

Pemakaian:
    python scripts/24_fix_composers.py
    python scripts/24_fix_composers.py --apply
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from laguqa.dataset.composers import main

if __name__ == "__main__":
    sys.exit(main())
