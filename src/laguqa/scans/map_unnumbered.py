#!/usr/bin/env python3
"""Beri nama berkas pindaian yang belum bernomor.

Dua puluh empat berkas pindaian belum memakai pola penamaan <id>_<judul>.jpg.
Isinya sudah diidentifikasi satu per satu dengan membaca lirik pada halaman dan
memeriksa silang tembusan cetak (bleed-through) dari halaman sebaliknya, yang
menampilkan judul lagu berikutnya sesuai urutan buku.

Berkas satu halaman yang ternyata punya halaman lanjutan ikut diberi akhiran _p1
supaya penomoran halaman konsisten.

Mode bawaan dry-run. Tambahkan --apply untuk menjalankan.

Pemakaian:
    python scripts/01_map_unnumbered.py
    python scripts/01_map_unnumbered.py --apply
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from laguqa.paths import RAW_SCANS_DIR as IMAGES

# berkas sumber -> (id lagu, nama tujuan, tingkat keyakinan, dasar identifikasi)
MAPPING: dict[str, tuple[int, str, str, str]] = {
    "img_20250626_111430.jpg": (55, "55_indonesia_raya_p2.jpg", "tinggi", "lirik 'bangunlah badannya untuk Indonesia Raya'"),
    "img_20250626_111526.jpg": (57, "57_hari_merdeka_p2.jpg", "tinggi", "lirik 'kita tetap setia mempertahankan Indonesia'"),
    "img_20250626_111706.jpg": (60, "60_maju_tak_gentar_p2.jpg", "tinggi", "lirik 'majulah majulah menang'; tembusan judul Berkibarlah Benderaku (61)"),
    "img_20250626_114334.jpg": (68, "68_rayuan_pulau_kelapa_p2.jpg", "tinggi", "lirik 'memuja pulau nan indah, tanah airku'; tembusan judul Rayuan Pulau Kelapa"),
    "img_20250626_114516.jpg": (69, "69_gugur_bunga_p2.jpg", "tinggi", "lirik 'gugur bungaku di taman bakti'"),
    "img_20250626_114832.jpg": (76, "76_bhineka_tunggal_ika_p2.jpg", "sedang", "akhir lagu 'Jiwa Indonesia bahagia'; tembusan judul Hymne Kemerdekaan (77) di halaman sebalik"),
    "img_20250626_114906.jpg": (77, "77_hymne_kemerdekaan_p2.jpg", "sedang", "hanya berisi diagram chord gitar; tembusan judul Hymne Pancasila (78) di halaman sebalik"),
    "img_20250626_115122.jpg": (83, "83_hymne_guru_p2.jpg", "tinggi", "lirik 'Engkau patriot pahlawan bangsa'; tembusan judul Hymne Guru"),
    "img_20250626_115148.jpg": (85, "85_indonesia_tumpah_darahku.jpg", "tinggi", "JUDUL TERBACA LANGSUNG: INDONESIA TUMPAH DARAHKU, Cipt. Ibu Sud"),
    "img_20250626_115441.jpg": (86, "86_mars_pon_p2.jpg", "tinggi", "lirik 'berlomba mengadu kekuatan tenaga untuk Nusa Bangsa'; tembusan judul Pemilu (87)"),
    "img_20250626_115521.jpg": (88, "88_eka_prasetya_pancakarsa.jpg", "tinggi", "JUDUL TERBACA LANGSUNG: EKA PRASETYA PANCAKARSA, Cipt. Sancaya HR"),
    "img_20250626_115724.jpg": (93, "93_bengawan_solo_p2.jpg", "tinggi", "lirik 'akhirnya ke laut, itu perahu riwayatmu dulu'"),
    "img_20250626_115745.jpg": (94, "94_surabaya_p2.jpg", "tinggi", "lirik 'Surabaya di tahun empat lima'; tembusan judul Surabaya"),
    "img_20250626_115905.jpg": (96, "96_teluk_bayur_p2.jpg", "tinggi", "lirik 'di Teluk Bayur'; tembusan judul Selendang Sutera (97)"),
    "img_20250626_115945.jpg": (98, "98_sepasang_mata_bola_p2.jpg", "tinggi", "lirik 'sepasang mata bola'; tembusan judul Jembatan Merah (99)"),
    "img_20250626_120018.jpg": (100, "100_bandung_selatan_di_waktu_malam_p2.jpg", "tinggi", "lirik 'Bandung Selatan di waktu malam'"),
    "img_20250626_120045.jpg": (101, "101_gubahanku_p2.jpg", "tinggi", "lirik 'setahun kita berpisah, sewindu terasa sudah'"),
    "img_20250626_120118.jpg": (102, "102_kopral_jono_p2.jpg", "tinggi", "lirik 'Kopral Jono' berulang; tembusan judul Kopral Jono"),
    "img_20250626_120138.jpg": (103, "103_juwita_malam_p2.jpg", "tinggi", "lirik 'juwita malam, dari bulankah tuan'"),
    "img_20250626_120200.jpg": (104, "104_kebyar_kebyar_p2.jpg", "tinggi", "lirik 'simponiku, Kebyar Kebyar'; tembusan judul Kebyar-Kebyar"),
    "img_20250626_120216.jpg": (105, "105_sapu_tangan_dari_bandung_selatan_p2.jpg", "tinggi", "lirik 'Bandung selatan jangan dilupakan'"),
    "img_20250626_120229.jpg": (106, "106_melati_di_tapal_batas_p2.jpg", "tinggi", "lirik 'duhai putri muda remaja'; tembusan judul Melati di Tapal Batas"),
    "a.jpg": (97, "97_selendang_sutera_p2.jpg", "tinggi", "lirik 'Selendang sutra, kini pembalut luka'; tembusan judul Sepasang Mata Bola (98)"),
    "pemilu_2.jpg": (87, "87_pemilu_p2.jpg", "tinggi", "tembusan judul Eka Prasetya Pancakarsa (88) di halaman sebalik"),
}

# lagu yang sebelumnya satu halaman dan kini punya halaman kedua
ADD_P1_SUFFIX = {
    "55_indonesia_raya.jpg": "55_indonesia_raya_p1.jpg",
    "57_hari_merdeka.jpg": "57_hari_merdeka_p1.jpg",
    "60_maju_tak_gentar.jpg": "60_maju_tak_gentar_p1.jpg",
    "68_rayuan_pulau_kelapa.jpg": "68_rayuan_pulau_kelapa_p1.jpg",
    "69_gugur_bunga.jpg": "69_gugur_bunga_p1.jpg",
    "76_bhineka_tunggal_ika.jpg": "76_bhineka_tunggal_ika_p1.jpg",
    "77_hymne_kemerdekaan.jpg": "77_hymne_kemerdekaan_p1.jpg",
    "83_hymne_guru.jpg": "83_hymne_guru_p1.jpg",
    "86_mars_pon.jpg": "86_mars_pon_p1.jpg",
    "87_pemilu.jpg": "87_pemilu_p1.jpg",
    "93_bengawan_solo.jpg": "93_bengawan_solo_p1.jpg",
    "94_surabaya.jpg": "94_surabaya_p1.jpg",
    "96_teluk_bayur.jpg": "96_teluk_bayur_p1.jpg",
    "97_selendang_sutera.jpg": "97_selendang_sutera_p1.jpg",
    "98_sepasang_mata_bola.jpg": "98_sepasang_mata_bola_p1.jpg",
    "100_bandung_selatan_di_waktu_malam.jpg": "100_bandung_selatan_di_waktu_malam_p1.jpg",
    "101_gubahanku.jpg": "101_gubahanku_p1.jpg",
    "102_kopral_jono.jpg": "102_kopral_jono_p1.jpg",
    "103_juwita_malam.jpg": "103_juwita_malam_p1.jpg",
    "104_kebyar_kebyar.jpg": "104_kebyar_kebyar_p1.jpg",
    "105_sapu_tangan_dari_bandung_selatan.jpg": "105_sapu_tangan_dari_bandung_selatan_p1.jpg",
    "106_melati_di_tapal_batas.jpg": "106_melati_di_tapal_batas_p1.jpg",
}

# metadata yang terbaca langsung dari halaman saat identifikasi
METADATA_FOUND = {
    85: {"title": "Indonesia Tumpah Darahku", "composer": "Ibu Sud", "key_signature": "C", "tempo": "Tempo cepat (♩=110)"},
    88: {"title": "Eka Prasetya Pancakarsa", "composer": "Sancaya HR", "key_signature": "C", "tempo": "Tempo lambat, Agung (♩=90)"},
}

STILL_MISSING = {62: "Bangun Pemudi Pemuda"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not IMAGES.is_dir():
        print(f"tidak ditemukan: {IMAGES}", file=sys.stderr)
        return 1

    present = {p.name for p in IMAGES.iterdir() if p.is_file()}
    print("MODE APPLY" if args.apply else "MODE DRY-RUN (tidak ada yang diubah)")

    plan: list[tuple[str, str, str]] = []

    print("\n--- Berkas belum bernomor -> nama baru ---")
    for src, (sid, dst, conf, why) in sorted(MAPPING.items(), key=lambda kv: kv[1][0]):
        mark = "!" if conf != "tinggi" else " "
        status = "" if src in present else "  [SUMBER TIDAK ADA]"
        print(f" {mark} id {sid:>3}  {src:28s} -> {dst}{status}")
        print(f"        dasar: {why}")
        if src in present:
            plan.append((src, dst, conf))

    print("\n--- Tambah akhiran _p1 pada halaman pertama ---")
    for src, dst in sorted(ADD_P1_SUFFIX.items(), key=lambda kv: int(kv[0].split("_")[0])):
        if src in present:
            print(f"   {src:46s} -> {dst}")
            plan.append((src, dst, "tinggi"))
        else:
            print(f"   {src:46s} -> [SUMBER TIDAK ADA]")

    targets = [d for _, d, _ in plan]
    dupes = {t for t in targets if targets.count(t) > 1}
    clash = {t for t in targets if t in present and t not in {s for s, _, _ in plan}}
    if dupes or clash:
        print("\n!! BENTROK, dibatalkan:")
        for t in sorted(dupes | clash):
            print(f"   {t}")
        return 1

    if args.apply:
        # dua tahap lewat nama sementara supaya tukar nama tidak saling menimpa
        for src, dst, _ in plan:
            shutil.move(str(IMAGES / src), str(IMAGES / f".tmp_{dst}"))
        for _, dst, _ in plan:
            shutil.move(str(IMAGES / f".tmp_{dst}"), str(IMAGES / dst))
        print(f"\n{len(plan)} berkas diubah namanya.")
    else:
        print(f"\n{len(plan)} berkas akan diubah. Jalankan dengan --apply.")

    print("\n--- Metadata yang terbaca langsung dari halaman ---")
    for sid, meta in METADATA_FOUND.items():
        print(f"  id {sid}: {meta}")

    print("\n--- Masih hilang, perlu difoto ulang ---")
    for sid, title in STILL_MISSING.items():
        print(f"  id {sid}: {title}")

    low = [(s, d) for s, d, c in plan if c != "tinggi"]
    if low:
        print("\n--- Keyakinan sedang, sebaiknya diperiksa manual ---")
        for s, d in low:
            print(f"  {s} -> {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
