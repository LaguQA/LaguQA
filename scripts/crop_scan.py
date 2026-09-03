#!/usr/bin/env python3
"""Helper - enlarge one system of a scanned page for close reading.

Not a workflow stage. Used when the validator flags a bar: enlarging the bar
is faster and more certain than asking the model to transcribe it again.
"""

import sys

from laguqa.scans.crop_scan import main

if __name__ == "__main__":
    sys.exit(main())
