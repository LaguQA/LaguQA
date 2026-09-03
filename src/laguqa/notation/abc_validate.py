#!/usr/bin/env python3
"""Validator notasi ABC 2.1 untuk dataset LaguQA.

Pemeriksaan dibagi dua tingkat, dan pembedaan ini disengaja.

TEMUAN (pelanggaran aturan, pasti salah)
  Konservasi ketukan: setiap bar harus berjumlah sesuai birama, termasuk
  perubahan birama inline [M:...]. Ditambah kelengkapan header, keseimbangan
  tanda kurung, konstruksi terlarang ABC 2.1, dan keselarasan lirik.

PERINGATAN (kejanggalan statistik, belum tentu salah)
  Pemeriksaan nada: ambitus, kepatuhan pada tangga nada, dan nada penutup.
  Diperlukan karena konservasi ketukan buta terhadap tinggi nada. Bar
  "g3 z2 c' c' c'" dan "G4 z c c c" sama-sama berjumlah 8/8, padahal yang
  pertama meleset satu oktaf. Kesalahan seperti ini hanya tertangkap lewat
  sebaran nada, bukan lewat durasi.

Keselarasan lirik diperiksa terhadap dua konvensi:
  - notes  : token w: hanya selaras ke not (konvensi ABC 2.1 yang benar)
  - events : token w: selaras ke not dan tanda istirahat (konvensi non-standar
             yang sering dipakai model saat mentranskripsi; dilaporkan terpisah)

Pemakaian:
    python scripts/06_validate_abc.py <file.abc> [file2.abc ...]
    python -m laguqa.notation.abc_validate --stdin < tune.abc
    python -m laguqa.notation.abc_validate --quiet data/abc/*.abc
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from fractions import Fraction

# --- Pola token -------------------------------------------------------------

HEADER_RE = re.compile(r"^([A-Za-z]):\s*(.*)$")
INLINE_FIELD_RE = re.compile(r"\[([A-Za-z]):([^\]]*)\]")
CHORD_RE = re.compile(r'"[^"]*"')
DECORATION_RE = re.compile(r"![^!]*!")
GRACE_RE = re.compile(r"\{[^}]*\}")
COMMENT_RE = re.compile(r"%.*$")

NOTE_RE = re.compile(
    r"(?P<acc>\^{1,2}|_{1,2}|=)?"
    r"(?P<letter>[A-Ga-gzxZ])"
    r"(?P<octave>[,']*)"
    r"(?P<length>\d+/\d+|\d+/|/\d+|/+|\d+)?"
)
TUPLET_RE = re.compile(r"\((?P<p>\d+)(?::(?P<q>\d*))?(?::(?P<r>\d*))?")

BARLINE_RE = re.compile(r"\|\]|::|:\|\]?|\|:|\|\||\[\||\|")
ENDING_RE = re.compile(r"\[[12](?:[,-]\d)*")

FORBIDDEN = {
    "dekorasi +...+ (usang)": re.compile(r"\+[a-z]+\+"),
    "field A: (usang, pakai O:)": re.compile(r"^A:", re.M),
    "Q: gaya lama tanpa satuan": re.compile(r"^Q:\s*\d+\s*$", re.M),
    "chord gaya +C+E+G+": re.compile(r"\+[A-G][#b]?\+"),
}

TUPLET_DEFAULT_Q = {2: 3, 3: 2, 4: 3, 5: 2, 6: 2, 7: 2, 8: 3, 9: 2}

# --- Nada --------------------------------------------------------------------

STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARP_ORDER = "FCGDAEB"
FLAT_ORDER = "BEADGCF"

# jumlah kres/mol untuk tonika mayor; minor dan modus digeser dari sini
MAJOR_ACC = {
    "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
    "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7,
}
# geseran modus terhadap mayor, dihitung dalam langkah lingkaran kuint
MODE_SHIFT = {
    "maj": 0, "ion": 0, "": 0,
    "min": -3, "m": -3, "aeo": -3,
    "mix": -1, "dor": -2, "phr": -4, "lyd": 1, "loc": -5,
}
# Ambang di bawah ini dikalibrasi pada tujuh lagu pertama, jadi masih sementara.
# Ambitus terukur pada transkripsi yang sudah diverifikasi: 5, 8, 12, 12, 13, 14
# semiton. Dua transkripsi yang meleset justru berada di 19 dan 29 semiton, dan
# keduanya terbukti salah oktaf. Batas 17 memisahkan kedua kelompok itu.
# Perlu dihitung ulang setelah 20 lagu terverifikasi.
AMBITUS_MAX = 17
LEAP_MAX = 12  # lompatan lebih dari satu oktaf antarnada praktis tidak ada
OFF_SCALE_MAX = 0.15
# Batas nada tengah, dikalibrasi ulang atas 107 transkripsi lengkap. Angka
# lama C5 diambil dari tiga belas berkas pertama yang hampir semuanya lagu
# daerah, dan ternyata terlalu ketat: lagu nasional beregister lebih tinggi
# (kuartil-3 nada tengahnya C5, lawan A4 pada lagu daerah). Ambang lama
# menandai 13 berkas, dan Berkibarlah Benderaku sudah dipastikan ke halaman
# bukunya benar - buku memang memberi titik oktaf naik pada 3 1 2 3 di sana.
#
# D5 menyisakan empat berkas bernada tengah E5 ke atas. Isyarat ini lemah dan
# tidak bisa dipertajam dari data yang ada: sebaran nada tengahnya menyambung
# dari C4 sampai G5 tanpa celah, jadi ambang mana pun memotong di tengah
# sebaran. Perlakukan hasilnya sebagai bahan periksa manual, bukan vonis.
MEDIAN_MAX = 74  # D5
MEDIAN_MIN = 55  # G3


def parse_key(val: str) -> tuple[int | None, dict[str, int]]:
    """Kembalikan (pitch class tonika, peta accidental tanda kunci)."""
    val = val.strip()
    if not val or val.lower() in {"none", "hp"}:
        return None, {}
    m = re.match(r"^([A-Ga-g])([#b]?)\s*([A-Za-z]*)", val)
    if not m:
        return None, {}
    letter = m.group(1).upper()
    sign = m.group(2)
    mode = m.group(3).lower()[:3]
    if mode in {"exp", "cle"}:
        mode = ""
    shift = MODE_SHIFT.get(mode)
    if shift is None:
        shift = 0
    n = MAJOR_ACC.get(letter + sign)
    if n is None:
        return None, {}
    n += shift
    acc: dict[str, int] = {}
    if n > 0:
        for i in range(min(n, 7)):
            acc[SHARP_ORDER[i]] = 1
    elif n < 0:
        for i in range(min(-n, 7)):
            acc[FLAT_ORDER[i]] = -1
    tonic = (STEP[letter] + (1 if sign == "#" else -1 if sign == "b" else 0)) % 12
    return tonic, acc


def note_to_midi(letter: str, octave_marks: str, acc_delta: int) -> int:
    base = STEP[letter.upper()]
    octave = 5 if letter.islower() else 4
    octave += octave_marks.count("'") - octave_marks.count(",")
    return (octave + 1) * 12 + base + acc_delta


def collect_pitches(text: str) -> list[int]:
    """Ambil semua nada bernada (bukan istirahat) sebagai nomor MIDI.

    Accidental yang ditulis eksplisit berlaku sampai akhir bar, sesuai ABC 2.1.
    """
    key_acc: dict[str, int] = {}
    in_body = False
    out: list[int] = []
    bar_acc: dict[str, int] = {}

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        m = HEADER_RE.match(line)
        if m and len(m.group(1)) == 1:
            if m.group(1) == "K" and not in_body:
                _, key_acc = parse_key(m.group(2))
                in_body = True
                continue
            if not in_body:
                continue
            if m.group(1) == "w":
                continue
        if not in_body:
            continue

        music = clean_music(line)
        for fm in INLINE_FIELD_RE.finditer(music):
            if fm.group(1) == "K":
                _, key_acc = parse_key(fm.group(2))
        music = INLINE_FIELD_RE.sub(" ", music)

        i = 0
        while i < len(music):
            if music[i] == "|":
                bar_acc = {}
                i += 1
                continue
            nm = NOTE_RE.match(music, i)
            if nm and nm.group("letter"):
                letter = nm.group("letter")
                if letter in "zxZ":
                    i = nm.end()
                    continue
                name = letter.upper()
                raw_acc = nm.group("acc")
                if raw_acc:
                    delta = 0 if raw_acc == "=" else (
                        len(raw_acc) if raw_acc[0] == "^" else -len(raw_acc)
                    )
                    bar_acc[name] = delta
                elif name in bar_acc:
                    delta = bar_acc[name]
                else:
                    delta = key_acc.get(name, 0)
                out.append(note_to_midi(letter, nm.group("octave") or "", delta))
                i = nm.end()
                continue
            i += 1
    return out


def midi_name(n: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[n % 12]}{n // 12 - 1}"


@dataclass
class BarReport:
    index: int
    line_no: int
    total: Fraction
    quota: Fraction
    text: str

    @property
    def ok(self) -> bool:
        return self.total == self.quota


@dataclass
class LyricReport:
    line_no: int
    tokens: int
    notes: int
    events: int

    @property
    def ok_notes(self) -> bool:
        return self.tokens == self.notes

    @property
    def ok_events(self) -> bool:
        return self.tokens == self.events

    @property
    def ok(self) -> bool:
        return self.ok_notes or self.ok_events


@dataclass
class TuneReport:
    title: str = ""
    meter: str = ""
    unit: str = ""
    key: str = ""
    bars: list[BarReport] = field(default_factory=list)
    lyrics: list[LyricReport] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    meter_changes: int = 0


def parse_length(raw: str | None) -> Fraction:
    """Ubah sufiks panjang ABC menjadi kelipatan unit L:."""
    if not raw:
        return Fraction(1)
    if set(raw) == {"/"}:
        return Fraction(1, 2 ** len(raw))
    if "/" in raw:
        num, _, den = raw.partition("/")
        return Fraction(int(num) if num else 1, int(den) if den else 2)
    return Fraction(int(raw))


def meter_to_quota(val: str) -> Fraction | None:
    val = val.strip()
    if val == "C":
        return Fraction(1)
    if val == "C|":
        return Fraction(1)
    if val.lower() == "none":
        return None
    if "/" in val:
        num, _, den = val.partition("/")
        try:
            # birama gabungan seperti 3+2/8
            total = sum(int(p) for p in num.split("+"))
            return Fraction(total, int(den))
        except ValueError:
            return None
    return None


def clean_music(line: str) -> str:
    """Buang komentar, grace note, dekorasi, dan chord. Field inline dipertahankan."""
    line = COMMENT_RE.sub("", line)
    line = GRACE_RE.sub("", line)
    line = DECORATION_RE.sub("", line)
    line = CHORD_RE.sub("", line)
    return line


def measure_bar(bar: str, unit: Fraction) -> tuple[Fraction, int, int]:
    """Kembalikan (durasi bar, jumlah not, jumlah not+istirahat)."""
    total = Fraction(0)
    notes = 0
    events = 0
    tuplet_left = 0
    tuplet_ratio = Fraction(1)
    i = 0
    while i < len(bar):
        ch = bar[i]

        if ch == "(":
            m = TUPLET_RE.match(bar, i)
            if m and m.group("p"):
                p = int(m.group("p"))
                q = int(m.group("q")) if m.group("q") else TUPLET_DEFAULT_Q.get(p, 2)
                r = int(m.group("r")) if m.group("r") else p
                tuplet_ratio = Fraction(q, p)
                tuplet_left = r
                i = m.end()
                continue
            i += 1
            continue

        if ch in ")- \t":
            i += 1
            continue

        m = NOTE_RE.match(bar, i)
        if m and m.group("letter"):
            length = parse_length(m.group("length")) * unit
            if tuplet_left > 0:
                length *= tuplet_ratio
                tuplet_left -= 1
                if tuplet_left == 0:
                    tuplet_ratio = Fraction(1)
            total += length
            events += 1
            if m.group("letter") not in "zxZ":
                notes += 1
            i = m.end()
            continue

        i += 1
    return total, notes, events


def count_lyric_tokens(w: str) -> int:
    """Hitung token w: yang memakan satu not: suku kata, '_' perpanjangan, '*' lewati."""
    w = COMMENT_RE.sub("", w)
    # Seluruh watak garis birama dibuang, bukan hanya "|". Model kerap
    # menyalin "|]" penutup lagu dan "[1" volta ke baris lirik, dan tanpa ini
    # sisa "]" atau "1" terhitung sebagai satu suku kata. Volta dibuang lebih
    # dahulu supaya angkanya ikut terbawa, bukan tertinggal sebagai token.
    w = re.sub(r"\|\]|\[\||\|?\[[12]", " ", w)
    w = re.sub(r"[|\[\]]", " ", w)
    w = w.replace(r"\-", "\x00")  # hyphen literal, bukan pemisah suku kata
    count = 0
    for chunk in w.split():
        for part in chunk.split("-"):
            if not part:
                continue
            core = part.strip("_")
            # Sisa tanda garis birama seperti ":" dari ":|" bukan suku kata.
            # "*" sebaliknya token sah yang memakan satu not.
            if core and core != "*" and not re.search(r"[\w\x00]", core):
                core = ""
            count += part.count("_") + (1 if core else 0)
    return count


def validate_tune(text: str) -> TuneReport:
    rep = TuneReport()
    unit = Fraction(1, 8)
    quota = Fraction(1)
    in_body = False
    last_counts = (0, 0)  # (notes, events) baris musik terakhir

    # Bar disimpan di luar perulangan baris karena di ABC 2.1 satu bar boleh
    # menyeberangi pergantian baris: baris yang tidak diakhiri "|" berlanjut
    # ke baris berikutnya. Menutup bar di tiap akhir baris membuat bar begitu
    # terhitung dua kali, keduanya kurang panjang.
    current = ""
    bar_line_no = 0

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("%"):
            continue

        m = HEADER_RE.match(line)
        if m and not in_body and len(m.group(1)) == 1:
            tag, val = m.group(1), m.group(2).strip()
            if tag == "T" and not rep.title:
                rep.title = val
            elif tag == "M":
                rep.meter = val
                q = meter_to_quota(val)
                if q:
                    quota = q
            elif tag == "L":
                rep.unit = val
                n, _, d = val.partition("/")
                unit = Fraction(int(n or 1), int(d or 8))
            elif tag == "K":
                rep.key = val
                in_body = True
            continue

        if not in_body:
            continue

        if m and m.group(1) == "w":
            tokens = count_lyric_tokens(m.group(2))
            rep.lyrics.append(LyricReport(line_no, tokens, last_counts[0], last_counts[1]))
            continue
        if m and m.group(1) in "XTCOQRNPZSIrsBDFGHUVWmU":
            continue

        music = clean_music(line)
        if not music.strip():
            continue

        # pisahkan menjadi segmen: field inline, barline, isi bar
        music = ENDING_RE.sub("|", music)
        pieces = re.split(r"(\[[A-Za-z]:[^\]]*\]|" + BARLINE_RE.pattern + ")", music)

        line_notes = line_events = 0
        for piece in pieces:
            if piece is None or piece == "":
                continue
            fm = INLINE_FIELD_RE.fullmatch(piece)
            if fm:
                tag, val = fm.group(1), fm.group(2)
                if tag == "M":
                    q = meter_to_quota(val)
                    if q:
                        quota = q
                        rep.meter_changes += 1
                elif tag == "L":
                    n, _, d = val.partition("/")
                    try:
                        unit = Fraction(int(n or 1), int(d or 8))
                    except ValueError:
                        pass
                continue
            if BARLINE_RE.fullmatch(piece):
                if current.strip():
                    total, _, _ = measure_bar(current, unit)
                    rep.bars.append(
                        BarReport(len(rep.bars) + 1, bar_line_no, total, quota, current.strip()[:58])
                    )
                current = ""
                continue
            # Not dihitung per potongan, bukan sewaktu bar ditutup, supaya
            # bar yang menyeberangi baris tetap menyumbangkan notnya ke baris
            # tempat not itu benar-benar tertulis. Penyelarasan lirik
            # dibandingkan per baris, jadi salah letak di sini menggeser
            # seluruh pemeriksaan lirik.
            _, notes, events = measure_bar(piece, unit)
            line_notes += notes
            line_events += events
            if not current.strip():
                bar_line_no = line_no
            current += piece

        last_counts = (line_notes, line_events)

    # Bar terakhir belum tentu ditutup garis birama, jadi ditutup di sini.
    if current.strip():
        total, _, _ = measure_bar(current, unit)
        rep.bars.append(
            BarReport(len(rep.bars) + 1, bar_line_no, total, quota, current.strip()[:58])
        )

    if not rep.key:
        rep.problems.append("header K: tidak ditemukan")
    if not rep.meter:
        rep.problems.append("header M: tidak ditemukan")
    if not rep.unit:
        rep.problems.append("header L: tidak ditemukan (default 1/8 dipakai)")
    if not text.lstrip().startswith("%abc"):
        rep.problems.append("baris pertama bukan %abc-2.1")
    return rep


def check_syntax(text: str) -> list[str]:
    out = [name for name, pat in FORBIDDEN.items() if pat.search(text)]
    # Baris lirik dibuang, baik w: yang selaras maupun W: yang dicetak sebagai
    # prosa di bawah lagu. Keduanya berisi tanda baca biasa: "(2x)" pada baris
    # W: pernah terhitung sebagai kurung tutup tanpa pembuka, karena pembukanya
    # diabaikan aturan triol "(3".
    body = "\n".join(
        COMMENT_RE.sub("", ln)
        for ln in text.splitlines()
        if not re.match(r"[wW]:", ln.lstrip())
    )
    if body.count('"') % 2:
        out.append("tanda kutip chord tidak berpasangan")
    opens = len(re.findall(r"\((?!\d)", body))
    closes = body.count(")")
    if opens != closes:
        out.append(f"slur tidak seimbang ({opens} buka, {closes} tutup)")

    # Garis birama ABC 2.1 memakai kurung siku tanpa pasangan: "|]" penutup
    # lagu, "[|" pembuka tebal, dan "[1" "[2" untuk volta. Ketiganya dibuang
    # dahulu supaya yang tersisa hanya kurung akor [CEG] dan field sisipan
    # [M:2/4], yang memang wajib berpasangan.
    siku = re.sub(r"\|\]|\[\||\|?\[[12]", "", body)
    if siku.count("[") != siku.count("]"):
        out.append("kurung siku tidak seimbang")
    return out


def check_pitch(text: str, key: str) -> tuple[list[str], dict]:
    """Periksa kewajaran sebaran nada. Hasilnya peringatan, bukan temuan.

    Konservasi ketukan tidak melihat tinggi nada sama sekali, sehingga
    kesalahan oktaf lolos begitu saja. Empat isyarat dipakai di sini: rentang
    nada yang terlalu lebar untuk suara manusia, letak melodi yang terlalu
    tinggi atau rendah secara keseluruhan, nada yang banyak keluar dari tangga
    nada, dan nada penutup yang bukan nada pokok.
    """
    pitches = collect_pitches(text)
    info: dict = {"n": len(pitches)}
    if not pitches:
        return [], info

    lo, hi = min(pitches), max(pitches)
    ambitus = hi - lo
    median = sorted(pitches)[len(pitches) // 2]
    info.update(lo=lo, hi=hi, ambitus=ambitus, median=median)
    warns: list[str] = []

    if ambitus > AMBITUS_MAX:
        warns.append(
            f"ambitus {ambitus} semiton ({midi_name(lo)}-{midi_name(hi)}), "
            f"lebih dari {AMBITUS_MAX}; periksa kemungkinan salah oktaf"
        )

    # Ambitus tidak berubah kalau seluruh lagu digeser satu oktaf, sehingga
    # kesalahan oktaf menyeluruh lolos dari pemeriksaan di atas. Letak melodi
    # diperiksa terpisah. Ambang diambil dari sebaran nada tengah pada dua
    # belas transkripsi yang registernya sudah dipastikan benar: D4 sampai B4,
    # dengan satu berkas di D5 yang memang bermasalah. Enam berkas hasil prompt
    # lama, yang seluruhnya naik satu oktaf, bernada tengah C5 sampai B5.
    if median > MEDIAN_MAX:
        warns.append(
            f"nada tengah {midi_name(median)} di atas {midi_name(MEDIAN_MAX)}; "
            "seluruh lagu kemungkinan tertulis satu oktaf terlalu tinggi"
        )
    elif median < MEDIAN_MIN:
        warns.append(
            f"nada tengah {midi_name(median)} di bawah {midi_name(MEDIAN_MIN)}; "
            "seluruh lagu kemungkinan tertulis satu oktaf terlalu rendah"
        )

    leaps = [(abs(b - a), i) for i, (a, b) in enumerate(zip(pitches, pitches[1:]), start=1)]
    big = [(d, i) for d, i in leaps if d > LEAP_MAX]
    info["max_leap"] = max((d for d, _ in leaps), default=0)
    if big:
        d, i = max(big)
        warns.append(
            f"{len(big)} lompatan lebih dari satu oktaf, terbesar {d} semiton "
            f"pada nada ke-{i} ({midi_name(pitches[i-1])} ke {midi_name(pitches[i])})"
        )

    tonic, key_acc = parse_key(key)
    if tonic is not None:
        scale = {(tonic + s) % 12 for s in (0, 2, 4, 5, 7, 9, 11)}
        off = [p for p in pitches if p % 12 not in scale]
        ratio = len(off) / len(pitches)
        info.update(off_scale=len(off), off_ratio=ratio)
        if ratio > OFF_SCALE_MAX:
            names = sorted({midi_name(p)[:-1] for p in off})
            warns.append(
                f"{len(off)}/{len(pitches)} nada ({ratio:.0%}) di luar tangga nada {key}: "
                f"{', '.join(names)}"
            )
        final = pitches[-1] % 12
        info["final"] = final
        if final == (tonic + 9) % 12:
            # Banyak lagu daerah ditulis "Do = C" tetapi berpusat pada la.
            # Tanda kunci tetap benar, yang berbeda hanya pusat nadanya.
            info["relative_minor"] = True
        elif final not in {tonic, (tonic + 7) % 12, (tonic + 4) % 12, (tonic + 2) % 12}:
            warns.append(
                f"nada penutup {midi_name(pitches[-1])} bukan nada pokok, kuint, terts, "
                f"atau la dari {key}"
            )
    return warns, info


def report(name: str, text: str, quiet: bool = False) -> tuple[int, dict]:
    if not text.strip():
        print(f"{name}: KOSONG")
        return 0, {}

    rep = validate_tune(text)
    syntax = rep.problems + check_syntax(text)
    pitch_warns, pitch_info = check_pitch(text, rep.key)

    interior = rep.bars[1:-1] if len(rep.bars) > 2 else []
    bad_bars = [b for b in interior if not b.ok]
    bad_lyrics = [l for l in rep.lyrics if not l.ok]
    events_only = [l for l in rep.lyrics if l.ok_events and not l.ok_notes]

    findings = len(syntax) + len(bad_bars) + len(bad_lyrics)

    stats = {
        "bars": len(rep.bars),
        "bad_bars": len(bad_bars),
        "lyric_lines": len(rep.lyrics),
        "bad_lyrics": len(bad_lyrics),
        "events_only": len(events_only),
        "syntax": len(syntax),
        "findings": findings,
        "warnings": len(pitch_warns),
    }

    if quiet:
        flag = "OK  " if findings == 0 else "GAGAL"
        print(
            f"{flag} {name:44s} bar {len(rep.bars)-len(bad_bars):>3}/{len(rep.bars):<3} "
            f"lirik {len(rep.lyrics)-len(bad_lyrics):>2}/{len(rep.lyrics):<2} "
            f"sintaks {len(syntax)}  temuan {findings}  peringatan {len(pitch_warns)}"
        )
        return findings, stats

    print(f"\n=== {name} ===")
    print(
        f"  T:{rep.title or '-'}  M:{rep.meter or '-'}  L:{rep.unit or '-'}  K:{rep.key or '-'}"
        + (f"  (+{rep.meter_changes} perubahan birama inline)" if rep.meter_changes else "")
    )

    for p in syntax:
        print(f"  [SINTAKS] {p}")

    if not rep.bars:
        print("  [BAR] tidak ada bar terbaca")
        return findings + 1, stats

    print(f"  [BAR] {len(rep.bars)} bar terbaca")
    if bad_bars:
        print(f"  [BAR] {len(bad_bars)} bar GAGAL konservasi ketukan:")
        for b in bad_bars[:12]:
            print(f"        baris {b.line_no:>3} bar#{b.index:<3} {b.total} != {b.quota}  |{b.text}|")
        if len(bad_bars) > 12:
            print(f"        ... dan {len(bad_bars)-12} lagi")
    else:
        print("  [BAR] semua bar interior LULUS")

    first, last = rep.bars[0], rep.bars[-1]
    if not first.ok or not last.ok:
        comp = "komplementer" if first.total + last.total == first.quota else "TIDAK komplementer"
        print(f"  [ANACRUSIS] awal {first.total} + akhir {last.total} -> {comp}")

    if rep.lyrics:
        if bad_lyrics:
            print(f"  [LIRIK] {len(bad_lyrics)}/{len(rep.lyrics)} baris w: tidak selaras:")
            for l in bad_lyrics[:8]:
                print(f"        baris {l.line_no:>3}: {l.tokens} token vs {l.notes} not / {l.events} not+istirahat")
            if len(bad_lyrics) > 8:
                print(f"        ... dan {len(bad_lyrics)-8} lagi")
        else:
            print(f"  [LIRIK] {len(rep.lyrics)} baris w: selaras")
        if events_only:
            print(
                f"  [LIRIK] {len(events_only)} baris memakai konvensi non-standar "
                "(token lirik diselaraskan ke tanda istirahat)"
            )

    if pitch_info.get("n"):
        print(
            f"  [NADA] {pitch_info['n']} nada, ambitus {pitch_info['ambitus']} semiton "
            f"({midi_name(pitch_info['lo'])}-{midi_name(pitch_info['hi'])}), "
            f"lompatan terbesar {pitch_info.get('max_leap', 0)} semiton"
            + (f", di luar tangga nada {pitch_info.get('off_scale', 0)}" if "off_scale" in pitch_info else "")
        )
    if pitch_info.get("relative_minor"):
        print(
            "  [NADA] berakhir pada la, jadi pusat nadanya minor relatif "
            "walau tanda kunci ditulis mayor"
        )
    for w in pitch_warns:
        print(f"  [PERINGATAN] {w}")

    print(f"  => TEMUAN: {findings}   PERINGATAN: {len(pitch_warns)}")
    return findings, stats


def main(argv: list[str]) -> int:
    quiet = "--quiet" in argv
    paths = [a for a in argv if not a.startswith("--")]

    if not paths or "--stdin" in argv:
        findings, _ = report("stdin", sys.stdin.read())
        return 0 if findings == 0 else 1

    total = 0
    agg = {
        "bars": 0, "bad_bars": 0, "lyric_lines": 0, "bad_lyrics": 0,
        "syntax": 0, "events_only": 0, "warnings": 0,
    }
    clean = 0
    n_stub = 0
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        # Berkas kosong buatan abc_stub.py dilewati, supaya satu glob bisa
        # dipakai sejak lagu pertama tanpa tenggelam dalam keluhan header.
        if not text.strip() or "% BELUM DIISI" in text.splitlines()[0]:
            n_stub += 1
            continue
        findings, stats = report(path, text, quiet=quiet)
        total += findings
        if findings == 0:
            clean += 1
        for k in agg:
            agg[k] += stats.get(k, 0)

    n = len(paths) - n_stub
    print(f"\n--- RINGKASAN {n} berkas ---")
    if n_stub:
        print(f"  {n_stub} berkas masih kosong, dilewati")
    print(f"  bersih tanpa temuan : {clean}/{n}")
    print(f"  bar lulus ketukan   : {agg['bars']-agg['bad_bars']}/{agg['bars']}")
    print(f"  baris lirik selaras : {agg['lyric_lines']-agg['bad_lyrics']}/{agg['lyric_lines']}")
    print(f"  konvensi lirik non-standar : {agg['events_only']} baris")
    print(f"  masalah sintaks     : {agg['syntax']}")
    print(f"  TOTAL TEMUAN        : {total}")
    print(f"  TOTAL PERINGATAN    : {agg['warnings']}  (kejanggalan nada, perlu diperiksa manual)")
    return 0 if total == 0 else 1


def cli() -> int:
    """Entry point for the console script; main() takes its argv explicitly."""
    return main(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
