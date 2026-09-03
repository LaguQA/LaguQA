#!/usr/bin/env python3
"""Stage 3 - rebuild data/laguqa.csv from the spreadsheet and the scan names.

Normalises page filenames, resolves numbering clashes, rebuilds
image_filename from what is actually on disk, and reports what still needs a
human decision.

Dry run by default; pass --apply to write.
"""

import sys

from laguqa.dataset.build import main

if __name__ == "__main__":
    sys.exit(main())
