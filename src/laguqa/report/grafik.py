"""Grafik analisis Bab IV, dibaca langsung dari tabel hasil dan manifes.

Penggunaan:
    python scripts/31_charts.py --keluar docs/gambar/bab4
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from laguqa.report.gaya import PT, PT_KECIL, SAMAR, TEKS, simpan

BIRU = "#5b7ea6"
JINGGA = "#c08a45"
HIJAU = "#6f8f79"
ABU = "#a9b2ba"
MERAH = "#a4553f"
KISI = "#e4e8ec"

KAMI = "gemma4-e2b"
KATEGORI = ["Fakta", "Tebak judul", "Lirik", "Notasi", "Penalaran"]


def kanvas(lebar_cm: float, tinggi_cm: float):
    fig = plt.figure(figsize=(lebar_cm / 2.54, tinggi_cm / 2.54))
    ax = fig.add_subplot(111)
    return fig, ax


def rapikan(ax, sumbu: str = "x") -> None:
    for sisi in ("top", "right"):
        ax.spines[sisi].set_visible(False)
    ax.spines["left" if sumbu == "x" else "bottom"].set_color("#c9d0d7")
    ax.spines["bottom" if sumbu == "x" else "left"].set_color("#c9d0d7")
    if sumbu == "x":
        ax.xaxis.grid(True, color=KISI, lw=0.6)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    else:
        ax.yaxis.grid(True, color=KISI, lw=0.6)
    ax.set_axisbelow(True)


def baca_papan(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def baca_luar(path: Path) -> dict[str, dict[str, float]]:
    keluar: dict[str, dict[str, float]] = {}
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if r["prompt"] not in ("netral", ""):
            continue
        keluar.setdefault(r["model"], {})[r["dataset"]] = float(r["akurasi"])
    return keluar


# --- Gambar 4.1 ------------------------------------------------------------

def skor_dasar(keluar: Path, papan: list[dict]) -> None:
    dasar = [r for r in papan if r["jenis"] == "dasar"]
    kontrol = {r["model"]: float(r["Keseluruhan"]) for r in papan
               if r["jenis"] == "kontrol"}
    dasar.sort(key=lambda r: float(r["Keseluruhan"]))

    fig, ax = kanvas(15.0, 9.6)
    nama = [r["model"] for r in dasar]
    nilai = [float(r["Keseluruhan"]) for r in dasar]
    warna = [JINGGA if n == KAMI else BIRU for n in nama]
    ax.barh(range(len(nama)), nilai, height=0.68, color=warna, edgecolor="none")

    for i, v in enumerate(nilai):
        ax.text(v + 0.4, i, f"{v:.1f}".replace(".", ","), va="center",
                ha="left", fontsize=PT_KECIL, color=SAMAR)

    lantai = kontrol.get("Tebakan ikut sebaran kunci", 32.1)
    acak = kontrol.get("Tebakan acak", 17.2)
    ax.axvline(lantai, color=MERAH, lw=1.0, ls="--", zorder=3)
    ax.axvline(acak, color=ABU, lw=1.0, ls=":", zorder=3)
    ax.text(lantai + 0.3, len(nama) - 0.2,
            f"tebakan ikut sebaran kunci {lantai:.1f}".replace(".", ","),
            color=MERAH, fontsize=PT_KECIL, va="center")
    ax.text(acak + 0.3, len(nama) - 1.2,
            f"tebakan acak {acak:.1f}".replace(".", ","),
            color=SAMAR, fontsize=PT_KECIL, va="center")

    ax.set_yticks(range(len(nama)))
    ax.set_yticklabels(nama, fontsize=PT_KECIL)
    ax.set_xlabel("Akurasi keseluruhan (persen)", fontsize=PT)
    ax.set_xlim(0, 40)
    ax.set_ylim(-0.8, len(nama) + 0.4)
    rapikan(ax, "x")
    fig.tight_layout()
    simpan(fig, keluar, "skor-dasar")


# --- Gambar 4.2 ------------------------------------------------------------

def pearson(x, y):
    return statistics.correlation(x, y) if len(x) > 2 else float("nan")


def sebaran_luar(keluar: Path, papan: list[dict], luar: dict) -> None:
    peta = {r["model"]: float(r["Keseluruhan"]) for r in papan
            if r["jenis"] == "dasar"}
    titik = []
    for nama, skor in peta.items():
        kunci = "base" if nama == KAMI else f"{nama}-base"
        if kunci in luar and "indommlu" in luar[kunci]:
            titik.append((luar[kunci]["indommlu"], skor, nama))
    titik.sort()

    x = [t[0] for t in titik]
    y = [t[1] for t in titik]
    r_semua = pearson(x, y)
    tanpa = [t for t in titik if t[2] != "sahabatai-9b"]
    r_tanpa = pearson([t[0] for t in tanpa], [t[1] for t in tanpa])

    fig, ax = kanvas(15.0, 10.4)
    ax.set_xlim(min(x) - 2.5, max(x) + 5.5)
    ax.set_ylim(16, 35)

    for xi, yi, nama in titik:
        ax.scatter(xi, yi, s=34, color=JINGGA if nama == KAMI else BIRU,
                   edgecolor="white", linewidth=0.7, zorder=4)

    lebar_x = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 15.0
    tinggi_y = (ax.get_ylim()[1] - ax.get_ylim()[0]) / 10.4
    dipakai: list[tuple[float, float, float, float]] = []
    for xi, yi, nama in sorted(titik, key=lambda t: -t[1]):
        w = len(nama) * 0.155 * lebar_x
        h = 0.42 * tinggi_y
        for dx, dy, ha in ((0.22, 0.30, "left"), (-0.22, 0.30, "right"),
                           (0.22, -0.55, "left"), (-0.22, -0.55, "right"),
                           (0.22, 0.85, "left"), (-0.22, -1.05, "right")):
            kx = xi + dx * lebar_x
            ky = yi + dy * tinggi_y
            x0 = kx if ha == "left" else kx - w
            kotak = (x0, ky - h / 2, x0 + w, ky + h / 2)
            if kotak[2] > ax.get_xlim()[1] or kotak[0] < ax.get_xlim()[0]:
                continue
            if any(not (kotak[2] < d[0] or kotak[0] > d[2]
                        or kotak[3] < d[1] or kotak[1] > d[3]) for d in dipakai):
                continue
            dipakai.append(kotak)
            ax.text(kx, ky, nama, fontsize=PT_KECIL - 0.5, color=SAMAR,
                    ha=ha, va="center")
            break

    ax.axhline(32.1, color=MERAH, lw=1.0, ls="--", zorder=2)
    ax.text(ax.get_xlim()[0] + 0.3, 32.4, "tebakan ikut sebaran kunci pada LaguQA",
            color=MERAH, fontsize=PT_KECIL, va="bottom")
    ax.axvline(23.4, color=ABU, lw=1.0, ls=":", zorder=2)
    ax.text(23.7, 16.4, "tebakan acak pada IndoMMLU", color=SAMAR,
            fontsize=PT_KECIL, rotation=90, va="bottom")

    ax.set_xlabel("Akurasi IndoMMLU (persen)", fontsize=PT)
    ax.set_ylabel("Akurasi LaguQA (persen)", fontsize=PT)
    catatan = (f"Pearson r = {r_semua:+.3f}".replace(".", ",") + "\n"
               + f"tanpa sahabatai-9b: r = {r_tanpa:+.3f}".replace(".", ","))
    ax.text(0.98, 0.04, catatan, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=PT_KECIL, color=TEKS,
            bbox=dict(boxstyle="round,pad=0.4", fc="#f7f9fb", ec="#dde3e9",
                      lw=0.6))
    rapikan(ax, "y")
    fig.tight_layout()
    simpan(fig, keluar, "sebaran-indommlu")
    return r_semua, r_tanpa


# --- Gambar 4.3 ------------------------------------------------------------

def kategori_latih(keluar: Path, papan: list[dict]) -> None:
    dasar = next(r for r in papan if r["model"] == KAMI and r["jenis"] == "dasar")
    latih = [r for r in papan if r["model"] == f"{KAMI} [lr4e4]"]
    if not latih:
        return

    fig, ax = kanvas(15.0, 8.6)
    kolom = ["Keseluruhan"] + KATEGORI
    x = range(len(kolom))
    sebelum = [float(dasar[k]) for k in kolom]
    nilai = {k: [float(r[k]) for r in latih] for k in kolom}
    rerata = [statistics.mean(nilai[k]) for k in kolom]
    bawah = [rerata[i] - min(nilai[k]) for i, k in enumerate(kolom)]
    atas = [max(nilai[k]) - rerata[i] for i, k in enumerate(kolom)]

    lebar = 0.36
    ax.bar([i - lebar / 2 for i in x], sebelum, lebar, color=ABU,
           edgecolor="none", label="Sebelum pelatihan")
    ax.bar([i + lebar / 2 for i in x], rerata, lebar, color=JINGGA,
           edgecolor="none", yerr=[bawah, atas], capsize=2.5,
           error_kw=dict(lw=0.8, ecolor=TEKS),
           label=f"Sesudah pelatihan, rerata {len(latih)} seed")

    for i, v in enumerate(sebelum):
        ax.text(i - lebar / 2, v + 1.2, f"{v:.0f}", ha="center", va="bottom",
                fontsize=PT_KECIL - 0.5, color=SAMAR)
    for i, v in enumerate(rerata):
        ax.text(i + lebar / 2, v + atas[i] + 1.2, f"{v:.0f}", ha="center",
                va="bottom", fontsize=PT_KECIL - 0.5, color=SAMAR)

    ax.set_xticks(list(x))
    ax.set_xticklabels(kolom, fontsize=PT_KECIL)
    ax.set_ylabel("Akurasi (persen)", fontsize=PT)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=PT_KECIL, loc="upper left", ncol=1)
    rapikan(ax, "y")
    fig.tight_layout()
    simpan(fig, keluar, "kategori-sebelum-sesudah")


# --- Gambar 4.4 ------------------------------------------------------------

def ragam_seed(keluar: Path, papan: list[dict]) -> None:
    varian = [("lr4e4", BIRU, -0.14), ("final", JINGGA, 0.14)]
    punya = [(v, w, d) for v, w, d in varian
             if len([r for r in papan if r["model"] == f"{KAMI} [{v}]"]) >= 2]
    if not punya:
        return
    kolom = ["Keseluruhan"] + KATEGORI

    fig, ax = kanvas(15.0, 8.6)
    ringkas = []
    for nama, w, geser in punya:
        latih = [r for r in papan if r["model"] == f"{KAMI} [{nama}]"]
        for i, k in enumerate(kolom):
            v = [float(r[k]) for r in latih]
            x = i + geser
            ax.plot([x, x], [min(v), max(v)], color=w, lw=1.0, alpha=0.45,
                    solid_capstyle="round", zorder=2)
            for nilai in v:
                ax.scatter(x, nilai, s=22, color=w, edgecolor="white",
                           linewidth=0.6, zorder=4)
            ax.scatter(x, statistics.mean(v), s=70, marker="_", color=w,
                       linewidth=1.8, zorder=5)
        semua = [float(r["Keseluruhan"]) for r in latih]
        ringkas.append(f"{nama}: rerata {statistics.mean(semua):.1f}, "
                       f"simpangan baku {statistics.stdev(semua):.2f}"
                       .replace(".", ","))
        ax.scatter([], [], s=26, color=w, label=f"varian {nama}")

    for i, k in enumerate(kolom):
        v = [float(r[k]) for nama, _, _ in punya
             for r in papan if r["model"] == f"{KAMI} [{nama}]"]
        ax.text(i, max(v) + 2.2, f"{max(v) - min(v):.1f}".replace(".", ","),
                ha="center", fontsize=PT_KECIL - 0.5, color=SAMAR)

    ax.set_xticks(range(len(kolom)))
    ax.set_xticklabels(kolom, fontsize=PT_KECIL)
    ax.set_ylabel("Akurasi (persen)", fontsize=PT)
    ax.set_xlim(-0.5, len(kolom) - 0.5)
    ax.set_title("Angka di atas tiap kelompok adalah jarak nilai tertinggi ke "
                 "terendah. " + "; ".join(ringkas) + ".",
                 fontsize=PT_KECIL, color=SAMAR, loc="left", pad=8)
    ax.legend(fontsize=PT_KECIL, loc="lower left", ncol=2)
    rapikan(ax, "y")
    fig.tight_layout()
    simpan(fig, keluar, "ragam-seed")


# --- Gambar 4.5 ------------------------------------------------------------

def lupa_slope(keluar: Path, luar: dict) -> None:
    tahap = [("base", "Sebelum\npelatihan"), ("final", "Varian\nfinal"),
             ("lr4e4", "Varian\nlr4e4")]
    ada = [t for t in tahap if t[0] in luar]
    if len(ada) < 2:
        return

    fig, ax = kanvas(11.5, 8.8)
    for nama, judul, warna in (("indommlu", "IndoMMLU", BIRU),
                               ("indoculture", "IndoCulture", JINGGA)):
        y = [luar[k][nama] for k, _ in ada if nama in luar[k]]
        x = list(range(len(y)))
        ax.plot(x, y, color=warna, lw=1.6, marker="o", ms=5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=4)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.1f}".replace(".", ","), (xi, yi),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=PT_KECIL, color=warna)
        ax.text(x[-1] + 0.08, y[-1], judul, color=warna, fontsize=PT_KECIL,
                va="center", ha="left", fontweight="bold")
        selisih = y[1] - y[0]
        ax.annotate(f"{selisih:+.1f} poin".replace(".", ","),
                    (0.5, (y[0] + y[1]) / 2), ha="center", va="center",
                    fontsize=PT_KECIL, color=SAMAR,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

    for nama, warna in (("indommlu", BIRU), ("indoculture", JINGGA)):
        if "acak" in luar and nama in luar["acak"]:
            ax.axhline(luar["acak"][nama], color=warna, lw=0.7, ls=":", alpha=0.6)

    ax.set_xticks(range(len(ada)))
    ax.set_xticklabels([j for _, j in ada], fontsize=PT_KECIL)
    ax.set_ylabel("Akurasi (persen)", fontsize=PT)
    ax.set_xlim(-0.35, len(ada) - 0.35)
    ax.set_ylim(20, 68)
    rapikan(ax, "y")
    fig.tight_layout()
    simpan(fig, keluar, "kemampuan-umum")


# --- Gambar 4.6 ------------------------------------------------------------

def kurva_loss(keluar: Path, manifes: list[Path]) -> None:
    if not manifes:
        return
    fig, ax = kanvas(15.0, 8.0)
    warna = [BIRU, JINGGA, HIJAU]
    for i, path in enumerate(sorted(manifes)):
        d = json.loads(path.read_text(encoding="utf-8"))
        sejarah = d.get("loss_history", [])
        if not sejarah:
            continue
        x = [h["step"] for h in sejarah if "step" in h]
        y = [h["loss"] for h in sejarah if "step" in h]
        if not x:
            x = list(range(1, len(sejarah) + 1))
            y = [h["loss"] for h in sejarah]
        ax.plot(x, y, color=warna[i % 3], lw=1.0, label=f"seed {d['seed']}")
        for e in d.get("eval_loss_history", []):
            ax.scatter(e["step"], e["eval_loss"], s=28, marker="D",
                       color=warna[i % 3], edgecolor="white", linewidth=0.7,
                       zorder=5)

    ax.set_xlabel("Langkah pelatihan", fontsize=PT)
    ax.set_ylabel("Loss", fontsize=PT)
    ax.set_yscale("log")
    ax.legend(fontsize=PT_KECIL, loc="upper right")
    ax.text(0.99, 0.72, "belah ketupat = validation loss akhir epoch",
            transform=ax.transAxes, ha="right", fontsize=PT_KECIL, color=SAMAR)
    rapikan(ax, "y")
    fig.tight_layout()
    simpan(fig, keluar, "kurva-loss")


# --- Gambar 4.7 ------------------------------------------------------------

def baca_prediksi(path: Path) -> dict[str, str]:
    keluar = {}
    for baris in path.open(encoding="utf-8"):
        d = json.loads(baris)
        if "prediksi" in d and "id" in d:
            keluar[d["id"]] = d["prediksi"]
    return keluar


def confusion_matrix(keluar: Path, kunci: dict[str, str], berkas: list[Path]) -> None:
    dasar = [p for p in berkas if "-base-" in p.name]
    latih = [p for p in berkas if "s1-lr4e4" in p.name]
    if not dasar:
        return
    huruf = ["A", "B", "C", "D", "E"]
    panel = [("Tiga belas model tanpa pelatihan", dasar)]
    if latih:
        panel.append(("Model sesudah pelatihan, seed 1", latih))

    fig, axes = plt.subplots(1, len(panel), figsize=(15.0 / 2.54, 8.2 / 2.54))
    if len(panel) == 1:
        axes = [axes]
    for ax, (judul, kelompok) in zip(axes, panel):
        m = [[0] * 5 for _ in range(5)]
        for path in kelompok:
            for ident, pilih in baca_prediksi(path).items():
                if ident in kunci and pilih in huruf:
                    m[huruf.index(kunci[ident])][huruf.index(pilih)] += 1
        total = sum(sum(b) for b in m) or 1
        persen = [[100 * v / total for v in b] for b in m]
        im = ax.imshow(persen, cmap="Blues", vmin=0, vmax=max(
            max(b) for b in persen))
        for i in range(5):
            for j in range(5):
                v = persen[i][j]
                ax.text(j, i, f"{v:.1f}".replace(".", ","), ha="center",
                        va="center", fontsize=PT_KECIL - 1,
                        color="white" if v > max(max(b) for b in persen) * 0.6
                        else TEKS)
        ax.set_xticks(range(5), huruf, fontsize=PT_KECIL)
        ax.set_yticks(range(5), huruf, fontsize=PT_KECIL)
        ax.set_xlabel("Huruf yang dipilih", fontsize=PT_KECIL)
        ax.set_ylabel("Huruf kunci", fontsize=PT_KECIL)
        ax.set_title(judul, fontsize=PT_KECIL, color=SAMAR, pad=6)
        for sisi in ax.spines.values():
            sisi.set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout()
    simpan(fig, keluar, "confusion-matrix")


# --- Gambar 4.8 ------------------------------------------------------------

def ukuran_skor(keluar: Path, papan: list[dict]) -> None:
    import re
    titik = []
    for r in papan:
        if r["jenis"] != "dasar":
            continue
        m = re.search(r"(\d+)b$", r["model"])
        if m:
            titik.append((int(m.group(1)), float(r["Keseluruhan"]), r["model"]))
    if len(titik) < 4:
        return

    fig, ax = kanvas(15.0, 8.6)
    for x, y, nama in titik:
        ax.scatter(x, y, s=34, color=JINGGA if nama == KAMI else BIRU,
                   edgecolor="white", linewidth=0.7, zorder=4)
    ax.axhline(32.1, color=MERAH, lw=1.0, ls="--", zorder=2)
    ax.text(0.4, 32.4, "tebakan ikut sebaran kunci", color=MERAH,
            fontsize=PT_KECIL, va="bottom")
    r = pearson([t[0] for t in titik], [t[1] for t in titik])
    ax.text(0.98, 0.05, f"Pearson r = {r:+.3f}".replace(".", ","),
            transform=ax.transAxes, ha="right", fontsize=PT_KECIL, color=TEKS,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f7f9fb", ec="#dde3e9",
                      lw=0.6))
    ax.set_xlabel("Jumlah parameter menurut nama model (miliar)", fontsize=PT)
    ax.set_ylabel("Akurasi LaguQA (persen)", fontsize=PT)
    ax.set_ylim(16, 35)
    rapikan(ax, "y")
    fig.tight_layout()
    simpan(fig, keluar, "ukuran-skor")


# --- Gambar 4.9 ------------------------------------------------------------

def kurva_checkpoint(keluar: Path, papan: list[dict]) -> None:
    peta = {r["model"]: r for r in papan}
    urutan = [(0, KAMI), (1309, f"{KAMI} [penuh-checkpoint-1311]"),
              (2618, f"{KAMI} [penuh-checkpoint-2622]")]
    ada = [(s, peta[n]) for s, n in urutan if n in peta]
    if len(ada) < 3:
        return

    fig, ax = kanvas(15.0, 8.2)
    kolom = ["Keseluruhan"] + KATEGORI
    warna = [TEKS, BIRU, JINGGA, HIJAU, MERAH, ABU]
    akhir = []
    for k, w in zip(kolom, warna):
        x = [s for s, _ in ada]
        y = [float(r[k]) for _, r in ada]
        ax.plot(x, y, color=w, lw=1.6 if k == "Keseluruhan" else 1.0,
                marker="o", ms=4, markeredgecolor="white", markeredgewidth=0.7,
                label=k)
        akhir.append([y[-1], k, w])

    akhir.sort(key=lambda a: -a[0])
    jarak = (max(a[0] for a in akhir) - min(a[0] for a in akhir)) * 0.075
    for i in range(1, len(akhir)):
        if akhir[i - 1][0] - akhir[i][0] < jarak:
            akhir[i][0] = akhir[i - 1][0] - jarak
    for y, k, w in akhir:
        ax.text(ada[-1][0] + 60, y, k, color=w, fontsize=PT_KECIL, va="center")
    ax.set_xlabel("Langkah pelatihan", fontsize=PT)
    ax.set_ylabel("Akurasi (persen)", fontsize=PT)
    ax.set_xlim(-120, 3400)
    ax.set_xticks([0, 1309, 2618], ["0\ntanpa latih", "1309\nsatu epoch",
                                    "2618\ndua epoch"], fontsize=PT_KECIL)
    rapikan(ax, "y")
    fig.tight_layout()
    simpan(fig, keluar, "kurva-checkpoint")


# --- Gambar 4.10 -----------------------------------------------------------

def kesulitan_butir(keluar: Path, kunci: dict[str, str], berkas: list[Path],
                    kategori: dict[str, str]) -> None:
    dasar = [p for p in berkas if "-base-" in p.name]
    if len(dasar) < 5:
        return
    benar = {ident: 0 for ident in kunci}
    for path in dasar:
        for ident, pilih in baca_prediksi(path).items():
            if kunci.get(ident) == pilih:
                benar[ident] += 1

    n = len(dasar)
    hitung = [0] * (n + 1)
    for v in benar.values():
        hitung[v] += 1

    fig, ax = kanvas(15.0, 8.0)
    warna = [MERAH if i == 0 else (HIJAU if i == n else BIRU)
             for i in range(n + 1)]
    ax.bar(range(n + 1), hitung, width=0.72, color=warna, edgecolor="none")
    for i, v in enumerate(hitung):
        if v:
            ax.text(i, v + 6, str(v), ha="center", va="bottom",
                    fontsize=PT_KECIL - 0.5, color=SAMAR)
    ax.set_xlabel(f"Jumlah model yang menjawab benar, dari {n} model tanpa pelatihan",
                  fontsize=PT)
    ax.set_ylabel("Jumlah soal", fontsize=PT)
    ax.set_xticks(range(n + 1))
    ax.set_xticklabels(range(n + 1), fontsize=PT_KECIL)
    mati = hitung[0]
    penuh = hitung[n]
    ax.set_title(f"{mati} soal tidak terjawab benar oleh satu model pun, "
                 f"{penuh} soal terjawab benar oleh semuanya.",
                 fontsize=PT_KECIL, color=SAMAR, loc="left", pad=8)
    rapikan(ax, "y")
    fig.tight_layout()
    simpan(fig, keluar, "kesulitan-butir")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keluar", type=Path, default=Path("docs/gambar/bab4"))
    ap.add_argument("--papan", type=Path,
                    default=Path("docs/tabel/mc/papan-skor.csv"))
    ap.add_argument("--luar", type=Path,
                    default=Path("docs/tabel/eksternal/akun1-l4/lupa.csv"))
    ap.add_argument("--hasil", type=Path, default=Path("hasil"))
    ap.add_argument("--mc", type=Path,
                    default=Path("data/benchmark/laguqa_mc.jsonl"))
    args = ap.parse_args(argv)

    papan = baca_papan(args.papan)
    luar = baca_luar(args.luar) if args.luar.is_file() else {}
    manifes = sorted(args.hasil.glob("*lr4e4-manifest.json"))
    prediksi = sorted(args.hasil.glob("*peluang-opsi--mc.jsonl"))
    kunci, kategori = {}, {}
    for baris in args.mc.open(encoding="utf-8"):
        d = json.loads(baris)
        if "kunci" in d:
            kunci[d["id"]] = d["kunci"]
            kategori[d["id"]] = d["kategori"]

    print(f"menulis ke {args.keluar}")
    skor_dasar(args.keluar, papan)
    confusion_matrix(args.keluar, kunci, prediksi)
    ukuran_skor(args.keluar, papan)
    kurva_checkpoint(args.keluar, papan)
    kesulitan_butir(args.keluar, kunci, prediksi, kategori)
    if luar:
        r = sebaran_luar(args.keluar, papan, luar)
        lupa_slope(args.keluar, luar)
        print(f"  korelasi LaguQA lawan IndoMMLU: r={r[0]:+.3f}, "
              f"tanpa sahabatai-9b r={r[1]:+.3f}")
    kategori_latih(args.keluar, papan)
    ragam_seed(args.keluar, papan)
    kurva_loss(args.keluar, manifes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
