#!/usr/bin/env python3
"""Stage 8 - fill the non-notation dataset columns from the ABC headers.

Composer, origin, tempo, time signature, key and lyrics were all read off the
page during transcription and already sit in the ABC header. Pulling them from
there avoids reading 107 images a second time.

Dry run by default; pass --apply to write.
"""

import sys

from laguqa.notation.abc_meta import main

if __name__ == "__main__":
    sys.exit(main())
