#!/usr/bin/env python3
"""Stage 16 - draw the thesis figures from the runs fetched into hasil/.

Writes a PDF and a PNG for each figure plus the CSV it was drawn from, so a
number in the text can be traced to a row and the figure can be rebuilt.

    python scripts/16_figures.py
    python scripts/16_figures.py --dir hasil --out docs/gambar
"""

import sys

from laguqa.report.figures import main

if __name__ == "__main__":
    sys.exit(main())
