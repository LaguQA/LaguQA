#!/usr/bin/env python3
"""Helper - shrink scanned pages, optionally to the title band only.

Not a workflow stage. Used when identifying which song an unnamed page holds.
"""

import sys

from laguqa.scans.make_thumbs import main

if __name__ == "__main__":
    sys.exit(main())
