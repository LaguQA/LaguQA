#!/usr/bin/env python3
"""Stage 10 - build the training and test sets from the frozen split.

The two sides are built from separate template pools, so the test score
measures song knowledge rather than template matching.

Dry run by default; pass --apply to write data/benchmark/.
"""

import sys

from laguqa.benchmark.generate import main

if __name__ == "__main__":
    sys.exit(main())
