#!/usr/bin/env python3
"""Stage 29 - rakit repositori model multi-varian dari folder rilis per run.

Varian yang ditaruh di akar adalah yang dimuat `PeftModel.from_pretrained`
tanpa `subfolder`, sehingga akar itulah rekomendasinya. Pilihannya diberikan
lewat --bawaan supaya tercatat di perintah, bukan di keadaan folder.

Dry run by default; pass --apply to write.

    python scripts/29_model_release.py --bawaan gemma4-e2b-full-s1-lr4e4
    python scripts/29_model_release.py --bawaan gemma4-e2b-full-s1-lr4e4 --apply
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from laguqa.report.model_release import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
