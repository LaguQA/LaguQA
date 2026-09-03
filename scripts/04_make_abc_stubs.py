#!/usr/bin/env python3
"""Stage 4 - lay out one empty .abc file per song, named from the dataset.

Transcription is pasted in one song at a time, and the filename is the only
record of which song the paste belongs to. Creating all the names up front
removes the chance of saving a transcription under the wrong number.

Dry run by default; pass --apply to create the files.
"""

import sys

from laguqa.notation.abc_stub import main

if __name__ == "__main__":
    sys.exit(main())
