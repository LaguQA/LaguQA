#!/usr/bin/env python3
"""Stage 1 - give the unnumbered raw scans their <id>_<title>.jpg names.

The photographed pages arrive with camera names like img_20250626_111430.jpg.
Until every page carries its song number, nothing downstream can find it.

Dry run by default; pass --apply to rename.
"""

import sys

from laguqa.scans.map_unnumbered import main

if __name__ == "__main__":
    sys.exit(main())
