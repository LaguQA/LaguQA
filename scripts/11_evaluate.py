#!/usr/bin/env python3
"""Stage 11 - score model answers against the LaguQA keys.

Takes a JSONL file of predictions (id_lagu, kategori, prediksi) and reports
strict and lenient accuracy side by side, per category.
"""

import sys

from laguqa.benchmark.evaluate import main

if __name__ == "__main__":
    sys.exit(main())
