#!/usr/bin/env python3
"""Stage 13 - add a title that identifies exactly one song.

Two songs in the book are both titled "Desaku", which made the same question
appear twice in the test set with opposing answers. This adds a title_unique
column for the question generators to use; the title column keeps whatever the
book prints.

Dry run by default; pass --apply to write laguqa.csv.
"""

import sys

from laguqa.dataset.titles import main

if __name__ == "__main__":
    sys.exit(main())
