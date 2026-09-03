#!/usr/bin/env python3
"""Stage 26 - build the external forgetting probes from IndoMMLU and IndoCulture.

Writes two LaguQA-MC shaped files to data/eksternal/ plus the frozen id list of
the IndoMMLU sample. Both sources are CC BY-NC-SA 4.0; see external.py for why
the output must not land in data/benchmark/.

Dry run by default; pass --apply to write.

    python scripts/26_external_bench.py
    python scripts/26_external_bench.py --apply
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from laguqa.benchmark import external as ex  # noqa: E402
from laguqa.paths import DATA_DIR  # noqa: E402


def unduh(repo: str, berkas: str) -> tuple[Path, str]:
    """Fetch one file from the Hub and report the revision it came from."""
    from huggingface_hub import hf_hub_download, repo_info
    revisi = repo_info(repo, repo_type="dataset").sha
    p = hf_hub_download(repo, berkas, repo_type="dataset", revision=revisi)
    return Path(p), revisi


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "eksternal")
    ap.add_argument("--sampel", type=int, default=2000,
                    help="jumlah soal IndoMMLU yang diambil")
    ap.add_argument("--benih", type=int, default=20260902)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args(argv)

    tanggal = dt.date.today().isoformat()
    hasil = {}

    repo, berkas = ex.REPOS["indommlu"]
    csv_mmlu, rev_mmlu = unduh(repo, berkas)
    skrip, _ = unduh(repo, "IndoMMLU.py")
    grup = ex.subject_groups(skrip)
    mmlu, buang_mmlu = ex.load_indommlu(csv_mmlu, grup)
    print(f"IndoMMLU  sumber {len(mmlu)} soal lolos, dibuang {buang_mmlu}")
    sampel = ex.stratified(mmlu, args.sampel, args.benih,
                           lambda x: (x["subjek"], x["tingkat"]))
    print(f"IndoMMLU  sampel {ex.ringkas(sampel)}")
    print(f"          {len({(x['subjek'], x['tingkat']) for x in sampel})} task, "
          f"{len({x['kategori'] for x in sampel})} grup")
    hasil["indommlu_mc.jsonl"] = (sampel, {
        "sumber": repo, "berkas": berkas, "revisi": rev_mmlu,
        "sha256_sumber": ex.sha256(csv_mmlu), "lisensi": ex.LISENSI,
        "sitasi": "Koto et al. 2023, IndoMMLU",
        "sampel": len(sampel), "populasi": len(mmlu),
        "strata": "subjek x tingkat", "benih": args.benih,
        "dibuang": buang_mmlu, "dibuat": tanggal,
        "catatan": "karya turunan CC BY-NC-SA 4.0, di luar rilis dataset LaguQA",
    })

    repo, berkas = ex.REPOS["indoculture"]
    csv_ic, rev_ic = unduh(repo, berkas)
    culture, buang_ic = ex.load_indoculture(csv_ic)
    print(f"IndoCulture  {ex.ringkas(culture)}, dibuang {buang_ic}")
    print(f"          {len({x['provinsi'] for x in culture})} provinsi, "
          f"{len({x['kategori'] for x in culture})} topik")
    hasil["indoculture_mc.jsonl"] = (culture, {
        "sumber": repo, "berkas": berkas, "revisi": rev_ic,
        "sha256_sumber": ex.sha256(csv_ic), "lisensi": ex.LISENSI,
        "sitasi": "Koto et al. 2024, IndoCulture",
        "sampel": len(culture), "populasi": len(culture),
        "strata": None, "benih": None,
        "dibuang": buang_ic, "dibuat": tanggal,
        "catatan": "karya turunan CC BY-NC-SA 4.0, di luar rilis dataset LaguQA",
    })

    if not args.apply:
        print("\ndry run, tidak ada yang ditulis. tambahkan --apply")
        return 0

    for nama, (items, header) in hasil.items():
        ex.write_mc(args.out / nama, header, items)
        print(f"ditulis {args.out / nama}")

    # The id list is published even though the questions are not. It is what
    # makes the sample checkable by someone who fetches IndoMMLU themselves, and
    # a list of identifiers carries none of the source text.
    daftar = args.out / "indommlu-sampel.json"
    daftar.write_text(json.dumps({
        "sumber": ex.REPOS["indommlu"][0], "revisi": rev_mmlu,
        "benih": args.benih, "strata": "subjek x tingkat",
        "id": [x["id"] for x in sampel],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"ditulis {daftar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
