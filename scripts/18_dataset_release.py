#!/usr/bin/env python3
"""Stage 18 - assemble the publishable dataset into a folder for the Hub.

Copies the benchmark, the table, the ABC transcriptions and SOURCE.md into one
directory with a hash manifest and a dataset card written from the data. The
scanned pages are never included; the book is still in copyright.

Dry run by default; pass --apply to write.

    python scripts/18_dataset_release.py
    python scripts/18_dataset_release.py --apply
"""

import sys

from laguqa.report.dataset_release import main

if __name__ == "__main__":
    sys.exit(main())
