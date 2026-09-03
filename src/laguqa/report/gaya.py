"""Gaya bersama seluruh gambar: ukuran huruf, warna, dan cara menyimpannya.

Dipakai dua pembangkit gambar yang tinggal di tempat berbeda. Grafik hasil
penelitian dibangkitkan `laguqa.report.grafik` dan ikut terbit bersama program
ini, sedangkan diagram alur metodologi dibangkitkan `penyusun.diagram` yang
hanya dipakai menyusun naskah laporan dan tidak diterbitkan. Keduanya harus
tampak berasal dari satu naskah, jadi gayanya ditetapkan sekali di sini.

Ukuran huruf sengaja lebih kecil daripada huruf badan naskah karena gambar
disisipkan dalam ukuran sentimeter yang pasti, tanpa penskalaan ulang oleh
pengolah kata.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

HURUF = ["Noto Sans", "Carlito", "Calibri", "DejaVu Sans"]

TEKS = "#1b1f24"
SAMAR = "#5c6670"
GARIS = "#39424c"

PT = 8.0
PT_KECIL = 7.0
PT_JUDUL = 8.5
CM_PER_PT = 2.54 / 72.0

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": HURUF,
    "font.size": PT,
    "text.color": TEKS,
    "axes.edgecolor": GARIS,
    "axes.labelcolor": TEKS,
    "xtick.color": SAMAR,
    "ytick.color": SAMAR,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def simpan(fig, keluar: Path, nama: str) -> None:
    """Simpan satu gambar sebagai png beresolusi cetak sekaligus pdf vektor."""
    keluar.mkdir(parents=True, exist_ok=True)
    for ext, dpi in (("png", 600), ("pdf", None)):
        fig.savefig(keluar / f"{nama}.{ext}", dpi=dpi, transparent=False,
                    facecolor="white")
    plt.close(fig)
    print(f"  {nama}.png dan {nama}.pdf")
