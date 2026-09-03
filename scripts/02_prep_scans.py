#!/usr/bin/env python3
"""Stage 2 - flatten the lighting on every raw page.

Phone photographs leave one half of the page darker than the other, which
buries the beam lines that carry note values. Flat-field correction turns the
raw pages into the ones actually sent to the vision model.

With no arguments this processes every .jpg in the raw scan directory into the
prepared scan directory. Any argument at all is passed straight through to
laguqa.scans.prep_scan instead.
"""

import sys

from laguqa.paths import RAW_SCANS_DIR, SCANS_DIR
from laguqa.scans.prep_scan import main


def run() -> int:
    if len(sys.argv) == 1:
        pages = sorted(str(p) for p in RAW_SCANS_DIR.glob("*.jpg"))
        if not pages:
            print(f"no .jpg found in {RAW_SCANS_DIR}", file=sys.stderr)
            return 1
        sys.argv += ["--out", str(SCANS_DIR)] + pages
    return main()


if __name__ == "__main__":
    sys.exit(run())
