#!/usr/bin/env python3
"""Attach a book page number to every song, then check the result for typos.

The page number is the only thing that lets a reader verify a transcription
against the printed source. The scans cannot be published, so "song 79 is on
page 127" is what stands in for them: an examiner who owns the book can open
that page and compare. A wrong page number quietly destroys that, because the
reader finds a different song there and concludes the dataset is unreliable.

Page numbers are typed by hand from 107 pages, so typos are expected rather
than hypothetical. Four properties catch almost all of them:

  order      songs are numbered in the order they appear in the book, so the
             page numbers must never go backwards
  span       a song marked as covering two sheets occupies page p and p+1,
             and its scan file list must have two entries to match
  overlap    two songs cannot start on the same page, and a two-page song
             cannot run into the next song's first page
  gap        pages nobody claims are reported but not treated as errors; a
             songbook legitimately has blank versos and section dividers

Gaps are the interesting output. Each one is either a real blank page or a
missed digit, and only the person holding the book can tell which.

On top of those, the numbers are checked against data/daftar-isi.csv, which is
the book's own table of contents typed out. That comparison is the one that
actually proves the column right, because it comes from the book rather than
from the same hand that filled the spreadsheet. It also lines the printed
titles up against ours, which is how the spelling of "Bhineka Tunggal Ika"
gets settled.

Reads halaman-buku-lengkap.csv from the workspace. Writes, with --apply:

  data/abc/*.abc     the "% laguqa-halaman" header, "127" or "159-160"
  data/laguqa.csv    columns book_page and book_page_end

Usage:
    python scripts/12_fill_pages.py            # check only
    python scripts/12_fill_pages.py --apply    # check, then write
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

from laguqa.paths import ABC_DIR, CSV_PATH, WORKSPACE

csv.field_size_limit(10**8)

PAGES_CSV = WORKSPACE / "halaman-buku-lengkap.csv"
TOC_PATH = CSV_PATH.parent / "daftar-isi.csv"
HEADER_RE = re.compile(r"^% laguqa-halaman .*$", re.MULTILINE)


def load_pages(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_songs() -> list[dict]:
    with open(CSV_PATH, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def parse(rows: list[dict], songs: list[dict]) -> tuple[list[dict], list[str]]:
    """Turn the spreadsheet into records, collecting anything unusable.

    Returns (records, fatal errors). A record has id, title, first page, last
    page, and how many scan files were listed for it.
    """
    by_id = {r["id"]: r for r in songs}
    errors: list[str] = []
    seen: set[str] = set()
    out: list[dict] = []

    for n, row in enumerate(rows, start=2):
        sid = (row.get("id") or "").strip()
        if not sid:
            errors.append(f"baris {n}: id kosong")
            continue
        if sid in seen:
            errors.append(f"baris {n}: id {sid} muncul dua kali")
            continue
        seen.add(sid)
        if sid not in by_id:
            errors.append(f"baris {n}: id {sid} tidak ada di laguqa.csv")
            continue

        raw = (row.get("halaman") or "").strip()
        if not raw or raw == "-":
            errors.append(f"baris {n}: lagu {sid} belum diisi halamannya")
            continue
        if not raw.isdigit():
            errors.append(f"baris {n}: halaman {raw!r} bukan angka")
            continue

        sheets = (row.get("lembar") or "1").strip()
        if sheets not in ("1", "2"):
            errors.append(f"baris {n}: lembar {sheets!r} bukan 1 atau 2")
            continue

        scans = [s for s in (row.get("berkas_pindaian") or "").split(",") if s.strip()]
        out.append(
            {
                "id": sid,
                "judul": by_id[sid]["title"],
                "awal": int(raw),
                "akhir": int(raw) + int(sheets) - 1,
                "lembar": int(sheets),
                "n_pindaian": len(scans),
            }
        )

    missing = sorted(set(by_id) - seen, key=int)
    if missing:
        errors.append(f"{len(missing)} lagu tidak ada di berkas halaman: {missing[:10]}")

    out.sort(key=lambda r: int(r["id"]))
    return out, errors


def check(records: list[dict]) -> tuple[list[str], list[str]]:
    """Returns (problems, notes). Problems block writing; notes do not."""
    problems: list[str] = []
    notes: list[str] = []

    # Order and overlap, walking the list in song order.
    for prev, cur in zip(records, records[1:]):
        if cur["awal"] <= prev["akhir"]:
            problems.append(
                f"lagu {cur['id']} ({cur['judul']}) mulai di halaman {cur['awal']}, "
                f"padahal lagu {prev['id']} ({prev['judul']}) masih memakai "
                f"halaman {prev['awal']}-{prev['akhir']}"
            )

    # A two-sheet song needs two scan files, a one-sheet song needs one.
    for r in records:
        if r["n_pindaian"] != r["lembar"]:
            problems.append(
                f"lagu {r['id']} ({r['judul']}) ditandai {r['lembar']} lembar "
                f"tapi punya {r['n_pindaian']} berkas pindaian"
            )

    # Gaps. Reported, never fatal: blank versos and dividers are normal.
    for prev, cur in zip(records, records[1:]):
        hole = cur["awal"] - prev["akhir"] - 1
        if hole > 0:
            span = (
                str(prev["akhir"] + 1)
                if hole == 1
                else f"{prev['akhir'] + 1}-{cur['awal'] - 1}"
            )
            notes.append(
                f"halaman {span} tidak diklaim lagu mana pun "
                f"(antara lagu {prev['id']} dan {cur['id']})"
            )

    return problems, notes


def loosen(title: str) -> str:
    """Strip everything that is only a matter of house style.

    The book prints titles in capitals with spaced hyphens; the dataset uses
    title case. Neither difference means the titles disagree, so both are
    flattened away and what remains is compared letter for letter.
    """
    return re.sub(r"[^a-z0-9]", "", title.lower())


def compare_toc(records: list[dict]) -> tuple[list[str], list[str]]:
    """Check the spreadsheet against the book's table of contents.

    Entries line up by position: the 55 Daerah songs are ids 0-54 and the 52
    Nasional songs are ids 55-106, in printed order. Returns (page mismatches,
    title mismatches); a page mismatch is fatal, a title mismatch is not.
    """
    if not TOC_PATH.exists():
        return ([f"tidak ada {TOC_PATH.name}, perbandingan daftar isi dilewati"], [])

    with open(TOC_PATH, encoding="utf-8") as fh:
        toc = list(csv.DictReader(fh))

    if len(toc) != len(records):
        return ([f"daftar isi {len(toc)} entri, lembar kerja {len(records)}"], [])

    pages: list[str] = []
    titles: list[str] = []
    for entry, r in zip(toc, records):
        if int(entry["halaman"]) != r["awal"]:
            pages.append(
                f"lagu {r['id']} ({r['judul']}): daftar isi halaman "
                f"{entry['halaman']}, lembar kerja {r['awal']}"
            )
        if loosen(entry["judul_buku"]) != loosen(r["judul"]):
            titles.append(
                f"lagu {r['id']}: buku menulis {entry['judul_buku']!r}, "
                f"dataset menulis {r['judul']!r}"
            )
    return pages, titles


def compare_headers(records: list[dict]) -> list[str]:
    """Find songs whose .abc already carries a different page number.

    Seven songs were filled in from the scans earlier. Where the spreadsheet
    disagrees with them, one of the two is wrong and neither can be assumed
    correct, so the disagreement is printed rather than silently overwritten.
    """
    out: list[str] = []
    for r in records:
        path = abc_path(r["id"])
        if path is None:
            continue
        m = re.search(r"^% laguqa-halaman (.+)$", path.read_text(encoding="utf-8"), re.M)
        if not m:
            continue
        old = m.group(1).strip()
        new = page_field(r)
        if old not in ("-", "", new):
            out.append(
                f"lagu {r['id']} ({r['judul']}): berkas .abc tertulis {old}, "
                f"lembar kerja tertulis {new}"
            )
    return out


def page_field(r: dict) -> str:
    return str(r["awal"]) if r["lembar"] == 1 else f"{r['awal']}-{r['akhir']}"


def abc_path(sid: str) -> Path | None:
    hits = sorted(ABC_DIR.glob(f"{int(sid):03d}_*.abc"))
    return hits[0] if hits else None


def write_abc(records: list[dict]) -> int:
    changed = 0
    for r in records:
        path = abc_path(r["id"])
        if path is None:
            print(f"  lewat: tidak ada berkas .abc untuk lagu {r['id']}")
            continue
        text = path.read_text(encoding="utf-8")
        line = f"% laguqa-halaman {page_field(r)}"
        new = HEADER_RE.sub(line, text, count=1)
        if new == text:
            continue
        path.write_text(new, encoding="utf-8")
        changed += 1
    return changed


def write_csv(records: list[dict], songs: list[dict]) -> None:
    by_id = {r["id"]: r for r in records}
    fields = list(songs[0].keys())
    for col in ("book_page", "book_page_end"):
        if col not in fields:
            fields.append(col)

    for row in songs:
        r = by_id.get(row["id"])
        row["book_page"] = str(r["awal"]) if r else ""
        row["book_page_end"] = str(r["akhir"]) if r else ""

    tmp = CSV_PATH.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(songs)
    tmp.replace(CSV_PATH)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="tulis ke .abc dan laguqa.csv")
    ap.add_argument("--sumber", type=Path, default=PAGES_CSV)
    args = ap.parse_args(argv)

    if not args.sumber.exists():
        print(f"tidak ada: {args.sumber}", file=sys.stderr)
        return 1

    songs = load_songs()
    records, errors = parse(load_pages(args.sumber), songs)

    if errors:
        print("=== isian tidak terbaca ===")
        for e in errors:
            print(f"  {e}")
        print()

    problems, notes = check(records)
    toc_pages, toc_titles = compare_toc(records)
    problems += toc_pages
    conflicts = compare_headers(records)

    print(f"lagu berhalaman: {len(records)}/{len(songs)}")
    if records:
        lo, hi = records[0]["awal"], records[-1]["akhir"]
        span = hi - lo + 1
        used = sum(r["lembar"] for r in records)
        double = sum(1 for r in records if r["lembar"] == 2)
        print(f"rentang halaman: {lo}-{hi}  ({span} halaman, {used} terpakai)")
        print(f"lagu dua halaman: {double}")

    print(f"cocok dengan daftar isi buku: {len(records) - len(toc_pages)}/{len(records)}")

    print(f"\nbentrok halaman / urutan mundur / lembar tidak cocok: {len(problems)}")
    for p in problems:
        print(f"  {p}")

    if toc_titles:
        print(f"\njudul yang ejaannya beda dari buku: {len(toc_titles)}")
        for t in toc_titles:
            print(f"  {t}")

    print(f"\nhalaman kosong (wajar, tapi periksa sekali): {len(notes)}")
    for n in notes:
        print(f"  {n}")

    if conflicts:
        print(f"\nbeda dengan yang sudah tercatat di .abc: {len(conflicts)}")
        for c in conflicts:
            print(f"  {c}")

    if errors or problems:
        print("\ntidak ditulis; perbaiki dulu yang di atas.")
        return 1

    if not args.apply:
        print("\npratinjau saja. tambahkan --apply untuk menulis.")
        return 0

    changed = write_abc(records)
    write_csv(records, songs)
    print(f"\nditulis: {changed} berkas .abc, dan {CSV_PATH.name} (+2 kolom)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
