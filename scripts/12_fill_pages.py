#!/usr/bin/env python3
"""Stage 12 - record which page of the book each song was transcribed from.

The scans are copyrighted and stay unpublished, so the page number is what a
reader uses to check a transcription against the printed original.

Dry run by default; pass --apply to write the .abc headers and laguqa.csv.
"""

import sys

from laguqa.dataset.pages import main

if __name__ == "__main__":
    sys.exit(main())
