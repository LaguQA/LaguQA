#!/usr/bin/env python3
"""Stage 17 - build the negative-control prediction files.

Writes three fake prediction files that the ordinary scorer reads: always the
most common answer, a random answer from the same category, and nothing at all.
Score them the same way as a model to see what each category is worth to
someone who knows no songs.

    python scripts/17_controls.py
    python scripts/11_evaluate.py hasil/kontrol-konstan--split70.jsonl
"""

import sys

from laguqa.benchmark.controls import main

if __name__ == "__main__":
    sys.exit(main())
