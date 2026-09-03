#!/usr/bin/env python3
"""Stage 15 - sambung kembali suku kata yang kehilangan tanda hubung di baris w:.

Kebalikan dari stage 14. Hanya memperbaiki pasangan yang dibuktikan korpus
sendiri, yaitu pasangan yang di tempat lain ditulis bertanda hubung.

Dry run; tambahkan --apply untuk menulis.
"""

import sys

from laguqa.notation.lyric_splits import main

if __name__ == "__main__":
    sys.exit(main())
