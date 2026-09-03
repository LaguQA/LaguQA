#!/usr/bin/env python3
"""Stage 7 - fold the validated .abc files into the dataset.

Writes abc_notation and abc_status. Only files the validator passes get
abc_status "terverifikasi", and only those may be used as answer keys.

Dry run by default; pass --apply to write.
"""

import sys

from laguqa.notation.abc_sync import main

if __name__ == "__main__":
    sys.exit(main())
