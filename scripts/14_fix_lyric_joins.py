#!/usr/bin/env python3
"""Stage 14 - unglue words that a stray capital joined in the lyric lines.

A capital transcribed onto a syllable that continues a word turns "In- do- ne-
sia" into "IndoNesia", and every lyric answer built from it inherits the fault.
Only the cases the corpus can settle are fixed; the rest are listed by page
number for someone holding the book.

Dry run by default; pass --apply to rewrite the .abc lyric lines.
"""

import sys

from laguqa.notation.lyric_joins import main

if __name__ == "__main__":
    sys.exit(main())
