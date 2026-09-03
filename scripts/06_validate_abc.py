#!/usr/bin/env python3
"""Stage 6 - run the ABC 2.1 validator over the transcriptions.

Reports findings (rule violations, certainly wrong) separately from warnings
(statistical oddities in the pitch distribution, not necessarily wrong).

With no file arguments this validates every .abc in data/abc/.
"""

import sys

from laguqa.notation.abc_validate import main
from laguqa.paths import ABC_DIR


def run() -> int:
    argv = sys.argv[1:]
    if not [a for a in argv if not a.startswith("--")]:
        argv += sorted(str(p) for p in ABC_DIR.glob("*.abc"))
    return main(argv)


if __name__ == "__main__":
    sys.exit(run())
