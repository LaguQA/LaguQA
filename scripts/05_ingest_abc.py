#!/usr/bin/env python3
"""Stage 5 - accept raw model output, check its contract, store it in data/abc/.

Unwraps the markdown fence, verifies the header contract, stamps the song id
and source image names from the input filename, then runs the validator.

Dry run by default; pass --apply to write.
"""

import sys

from laguqa.notation.abc_ingest import main

if __name__ == "__main__":
    sys.exit(main())
