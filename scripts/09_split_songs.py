#!/usr/bin/env python3
"""Stage 9 - split the songs into a train side and a test side, then freeze it.

The split is per song and happens before any question is generated; splitting
afterwards would put variants of the same song on both sides.

Dry run by default; pass --apply to freeze data/split.json.
"""

import sys

from laguqa.dataset.split import main

if __name__ == "__main__":
    sys.exit(main())
