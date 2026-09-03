#!/usr/bin/env python3
"""Stage 27 - table the forgetting probes against the base model.

Dry run by default; pass --apply to write docs/tabel/eksternal/lupa.md.

    python scripts/27_external_report.py
    python scripts/27_external_report.py --apply
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from laguqa.report.external import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
