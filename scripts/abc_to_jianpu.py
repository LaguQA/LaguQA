#!/usr/bin/env python3
"""Helper - convert ABC back to jianpu (notasi angka).

Not a workflow stage. Used two ways: to render the notation the benchmark
questions actually show, and as an independent cross-check that a
transcription matches the digits printed on the scanned page.
"""

import sys

from laguqa.notation.abc_to_jianpu import main

if __name__ == "__main__":
    sys.exit(main())
