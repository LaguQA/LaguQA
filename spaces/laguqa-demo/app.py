"""LaguQA: demonstrasi benchmark lagu nasional dan lagu daerah Indonesia."""

from __future__ import annotations

import html
import json
import math
import os
import random
from pathlib import Path

import gradio as gr

try:
    import spaces
except ImportError:  # supaya app.py tetap bisa dijalankan di luar Space
    class spaces:  # type: ignore
        @staticmethod
        def GPU(*a, **k):
            return (lambda f: f) if not a else a[0]


DATA = Path(__file__).parent / "data"
BASE_ID = os.environ.get("LAGUQA_BASE", "google/gemma-4-E2B-it")
ADAPTER_ID = os.environ.get("LAGUQA_ADAPTER", "IRedDragonICY/LaguQA-Gemma4-E2B")
SISTEM = ("Kamu asisten yang menguasai lagu nasional dan lagu daerah Indonesia, "
          "termasuk notasi angka dan notasi ABC-nya.")

LAGU = json.loads((DATA / "lagu.json").read_text(encoding="utf-8"))
SOAL = json.loads((DATA / "soal.json").read_text(encoding="utf-8"))
RINGKAS = json.loads((DATA / "ringkas.json").read_text(encoding="utf-8"))
JUDUL = {f"{s['judul']} · hal. {s['halaman']}": s for s in LAGU}
URUT_BUKU = "Urutan buku"
URUT_ABJAD = "Abjad"


def nomor_halaman(nama: str) -> int:
    angka = "".join(c for c in str(JUDUL[nama]["halaman"]) if c.isdigit())
    return int(angka) if angka else 10**6


def urutkan(nama: list[str], urutan: str) -> list[str]:
    """Urutan buku memakai nomor halaman, bukan id, supaya cocok dengan cetakan."""
    if urutan == URUT_BUKU:
        return sorted(nama, key=lambda n: (nomor_halaman(n), n))
    return sorted(nama)


NAMA = urutkan(list(JUDUL), URUT_BUKU)
MENTAH = RINGKAS["lagu"] - RINGKAS["terverifikasi"]
KATEGORI = sorted({x["kategori"] for x in SOAL})
SEMUA = "Semua kategori"

# Nomor program General MIDI. Piano dipakai sebagai bawaan karena serangannya
# tegas, sehingga salah durasi terdengar jelas ketika transkripsi diperiksa.
ALAT = {"Piano": 0, "Gitar nilon": 24, "Seruling": 73, "Biola": 40,
        "Kotak musik": 10, "Paduan suara": 52}
ABCJS = "6.7.0"

# --- model -------------------------------------------------------------------
#
# Ditempatkan ke cuda di tingkat modul, sesuai anjuran ZeroGPU. Di luar fungsi
# berdekorator, PyTorch berjalan dalam mode emulasi CUDA sehingga penempatan ini
# berhasil meski GPU sesungguhnya baru dialokasikan saat fungsi dipanggil.
# Memuat model di dalam fungsi berdekorator justru jauh lebih lambat.
#
# Seluruhnya dibungkus try supaya tiga tab lainnya tetap hidup ketika bobotnya
# belum terbit atau token akses model dasar belum dipasang. Tanpa ini, satu
# repositori yang belum ada membuat Space gagal start dan tidak menampilkan
# apa pun, termasuk bagian yang tidak membutuhkan GPU sama sekali.

tokenizer = None
model = None
GALAT_MODEL = ""

try:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    model = AutoModelForCausalLM.from_pretrained(BASE_ID, dtype=torch.bfloat16)
    if ADAPTER_ID:
        # torch_device="cpu" wajib di ZeroGPU. Tanpa itu PEFT memanggil
        # infer_device(), yang membaca torch.cuda.is_available() sebagai True
        # karena di luar fungsi berdekorator ada emulasi CUDA, lalu menyuruh
        # safetensors memuat bobot adapter langsung ke perangkat CUDA. Jalur
        # muat-langsung itu menyentuh CUDA sungguhan di proses utama, yang belum
        # memegang GPU, dan gagal dengan "No CUDA GPUs are available" dari dalam
        # safetensors. Bobot dasarnya sendiri lolos karena transformers memuat
        # ke CPU lebih dulu. Yang boleh menyentuh CUDA hanya .to() di bawah,
        # sebab itulah satu-satunya yang dicegat spaces.
        try:
            model = PeftModel.from_pretrained(model, ADAPTER_ID,
                                              torch_device="cpu")
        except TypeError:  # peft lama belum menerima torch_device
            model = PeftModel.from_pretrained(model, ADAPTER_ID)
    model.to("cuda")
    model.eval()
except Exception as exc:  # noqa: BLE001
    import traceback
    GALAT_MODEL = f"{type(exc).__name__}: {exc}".strip() or type(exc).__name__
    model = None
    # Dicetak utuh ke stdout supaya masuk log Space, dan ringkasannya nanti
    # ikut ditampilkan di antarmuka. Versi sebelumnya menyimpan pesan ini lalu
    # tidak pernah menunjukkannya, sehingga yang terbaca pengunjung hanya
    # "bobot belum terpasang" tanpa sebab, dan yang terbaca penulisnya cuma
    # nama kelas galat pada toast Gradio.
    print("=" * 70)
    print("GAGAL MEMUAT MODEL")
    traceback.print_exc()
    print("=" * 70, flush=True)


def lama_generasi(pesan, riwayat, temperature, maks_token) -> int:
    """Durasi GPU yang diminta, diperkirakan dari panjang jawaban.

    Durasi yang lebih pendek menaikkan prioritas antrean pengunjung, sehingga
    meminta 60 detik untuk jawaban 64 token akan merugikan pemakai lain.
    """
    return int(15 + int(maks_token) * 0.06)


def hasilkan(turns: list[dict], temperature: float, maks_token: int) -> str:
    """Satu giliran generasi, dengan galat yang bisa dibaca kalau gagal.

    Pembungkus try-nya ada karena kegagalan di ZeroGPU sering muncul sebagai
    RuntimeError berpesan kosong, dan Gradio kemudian hanya menampilkan nama
    kelasnya. Pemberitahuan bertuliskan "RuntimeError" saja tidak memberi tahu
    siapa pun apa yang rusak, sedangkan log Space belum tentu bisa dibuka
    orang yang melaporkannya.
    """
    prompt = tokenizer.apply_chat_template(turns, tokenize=False,
                                           add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    try:
        with torch.no_grad():
            keluar = model.generate(
                **enc, max_new_tokens=int(maks_token),
                do_sample=temperature > 0,
                temperature=max(float(temperature), 1e-5),
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    except Exception as exc:  # noqa: BLE001
        import traceback
        jejak = traceback.format_exc().strip().splitlines()
        pesan = str(exc).strip() or "(galat tanpa pesan)"
        raise gr.Error(f"Generasi gagal. {type(exc).__name__}: {pesan} "
                       f"[{jejak[-2].strip() if len(jejak) > 1 else ''}] "
                       f"perangkat model={getattr(model, 'device', '?')}, "
                       f"dtype={next(model.parameters()).dtype}") from exc
    baru = keluar[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(baru, skip_special_tokens=True).strip()


@spaces.GPU(duration=lama_generasi)
def jawab(pesan: str, riwayat: list, temperature: float,
          maks_token: int) -> str:
    if model is None:
        raise gr.Error("Model belum tersedia di Space ini. " + GALAT_MODEL)

    turns = [{"role": "system", "content": SISTEM}]
    for item in riwayat:
        if isinstance(item, dict):
            turns.append({"role": item["role"], "content": item["content"]})
        else:
            turns += [{"role": "user", "content": item[0]},
                      {"role": "assistant", "content": item[1]}]
    turns.append({"role": "user", "content": pesan})
    return hasilkan(turns, temperature, maks_token)


def lama_banding(pesan, riwayat_dasar, riwayat_latih, temperature,
                 maks_token) -> int:
    return int(20 + int(maks_token) * 0.12)


@spaces.GPU(duration=lama_banding)
def bandingkan(pesan: str, riwayat_dasar: list, riwayat_latih: list,
               temperature: float, maks_token: int) -> tuple[list, list, str]:
    """Dua percakapan berjalan bersamaan atas pertanyaan yang sama.

    Satu model di memori, bukan dua. `disable_adapter()` mematikan bobot LoRA
    untuk sementara, sehingga jawaban sisi kiri benar-benar keluar dari Gemma
    yang belum dilatih, dari berkas bobot yang sama, tanpa mengunduh atau
    menyimpan salinan kedua. Memuat dua model utuh akan melipatduakan VRAM dan
    waktu muatnya, dan pada ZeroGPU itu berarti gagal start.

    Tiap sisi meneruskan riwayatnya sendiri. Itu yang membuat ini dua
    percakapan dan bukan satu: setelah giliran pertama jawaban keduanya
    berbeda, dan memberi satu sisi jawaban milik sisi lain berarti memintanya
    melanjutkan kalimat yang tidak pernah ia ucapkan.

    Keduanya juga menerima system prompt yang sama. Memberi prompt tentang lagu
    hanya kepada satu sisi akan mengukur selisih label, bukan selisih
    pengetahuan, dan itu kekeliruan yang sama dengan yang dijaga uji dua kondisi
    prompt pada percobaan lupa.

    Satu panggilan GPU untuk dua jawaban, bukan dua panggilan. Kuota ZeroGPU
    dihitung per panggilan berdurasi, jadi memisahkannya akan membebani
    pengunjung dua kali antrean.
    """
    riwayat_dasar = list(riwayat_dasar or [])
    riwayat_latih = list(riwayat_latih or [])
    tanya = (pesan or "").strip()
    if not tanya:
        return riwayat_dasar, riwayat_latih, ""
    if model is None:
        raise gr.Error("Model belum tersedia di Space ini. " + GALAT_MODEL)

    def percakapan(riwayat: list) -> list[dict]:
        return ([{"role": "system", "content": SISTEM}] + riwayat
                + [{"role": "user", "content": tanya}])

    dilatih = hasilkan(percakapan(riwayat_latih), temperature, maks_token)
    if hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            dasar = hasilkan(percakapan(riwayat_dasar), temperature, maks_token)
    else:
        dasar = "(Space ini berjalan tanpa adapter, jadi tidak ada pembanding.)"

    riwayat_dasar += [{"role": "user", "content": tanya},
                      {"role": "assistant", "content": dasar}]
    riwayat_latih += [{"role": "user", "content": tanya},
                      {"role": "assistant", "content": dilatih}]
    return riwayat_dasar, riwayat_latih, ""


# --- potongan tampilan -------------------------------------------------------

def ribuan(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def peringatan_model() -> str:
    """Spanduk saat bobot gagal dimuat, berikut sebabnya.

    Sebabnya ikut ditampilkan karena tanpa itu pengunjung dan penulisnya
    sama-sama hanya melihat "belum terpasang", padahal galatnya sudah
    tertangkap sejak start. Menebak dari gejala saja sudah dua kali meleset.
    """
    return ('<div class="lq-peringatan">Bobot model belum terpasang di Space '
            'ini, sehingga tab Percakapan dan Bandingkan jawaban belum bisa '
            'dipakai. Empat tab lainnya berjalan tanpa model.'
            f'<br><code>{html.escape(GALAT_MODEL)}</code></div>')


def keping(teks: str, jenis: str = "") -> str:
    kelas = f"lq-keping {jenis}".strip()
    return f'<span class="{kelas}">{html.escape(teks)}</span>'


def bar_notasi(notasi: str) -> str:
    """Notasi angka dipecah per bar supaya garis biramanya terbaca sebagai kolom."""
    bar = [b.strip() for b in (notasi or "").split("|") if b.strip()]
    if not bar:
        return '<div class="lq-notasi lq-kosong">Notasi lagu ini belum tersedia.</div>'
    isi = "".join(f"<span>{html.escape(b)}</span>" for b in bar)
    return f'<div class="lq-notasi">{isi}</div>'


# --- katalog lagu ------------------------------------------------------------

def saring(cari: str, jenis: str, urutan: str, sekarang: str | None = None):
    kata = (cari or "").strip().lower()
    cocok = urutkan([n for n, s in JUDUL.items()
                     if (jenis == "Semua" or s["jenis"] == jenis)
                     and (not kata or kata in n.lower()
                          or kata in (s["pencipta"] or "").lower()
                          or kata in (s["asal"] or "").lower())], urutan)
    # Lagu yang sedang dibuka dipertahankan bila masih lolos saringan, supaya
    # sekadar mengubah urutan tidak melompat ke lagu lain.
    pilih = sekarang if sekarang in cocok else (cocok[0] if cocok else None)
    return gr.update(choices=cocok, value=pilih)


def kartu_lagu(judul: str):
    """Kartu metadata, notasi angka, dan sumber ABC untuk satu lagu.

    Keluaran ketiga adalah teks ABC mentah. Nilainya masuk ke kotak
    tersembunyi, lalu dibaca abcjs di sisi peramban untuk menggambar not
    baloknya dan membunyikannya.
    """
    s = JUDUL.get(judul)
    if not s:
        return ('<div class="lq-kartu lq-kosong">Tidak ada lagu yang cocok '
                'dengan penyaringnya.</div>', "", "")

    terverifikasi = s["abc_status"] == "terverifikasi"
    tanda = [keping(s["jenis"], "lq-jenis"),
             keping("notasi terverifikasi" if terverifikasi else "notasi mentah",
                    "lq-ok" if terverifikasi else "lq-awas")]

    baris = [("Nada dasar", f"Do = {s['nada_dasar']}"),
             ("Birama", s["birama"]),
             ("Tempo", s["tempo"]),
             ("Pencipta", s["pencipta"] or "tidak tercantum"),
             ("Daerah asal", s["asal"] or "tidak tercantum"),
             ("Halaman buku", s["halaman"])]
    isi = "".join(
        f"<dt>{html.escape(k)}</dt><dd>{html.escape(str(v))}</dd>"
        for k, v in baris)

    catatan = ""
    if not terverifikasi:
        catatan = ('<p class="lq-catatan-kartu">Notasi lagu ini belum lolos '
                   'pemeriksaan konservasi ketukan dan keselarasan lirik, '
                   'sehingga sebagian nadanya dapat terbaca dan terdengar '
                   'keliru.</p>')

    kartu = (f'<div class="lq-kartu">'
             f'<div class="lq-kepala-kartu"><h3>{html.escape(s["judul"])}</h3>'
             f'<div class="lq-tanda">{"".join(tanda)}</div></div>'
             f'<dl>{isi}</dl>{catatan}</div>')
    return kartu, bar_notasi(s["notasi"]), s["abc"]


# --- soal --------------------------------------------------------------------

def soal_html(x: dict, i: int) -> str:
    bagian = x["pertanyaan"].split("\n\n", 1)
    # Nomor urut dan id resmi dicetak berdampingan supaya penguji yang membuka
    # berkas soalnya dapat mencari baris yang sama tanpa menghitung sendiri.
    tanda = (keping(f"soal {i + 1} dari {len(SOAL)}")
             + keping(x["id"], "lq-kat")
             + keping(x["kategori"], "lq-kat")
             + keping(x["tingkat"]))
    badan = f'<p class="lq-tanya">{html.escape(bagian[0])}</p>'
    if len(bagian) > 1:
        badan += bar_notasi(bagian[1]) if "|" in bagian[1] else (
            f'<div class="lq-kutipan">{html.escape(bagian[1])}</div>')
    return f'<div class="lq-soal"><div class="lq-tanda">{tanda}</div>{badan}</div>'


def kandidat(kategori: str) -> list[int]:
    cocok = [i for i, x in enumerate(SOAL)
             if kategori == SEMUA or x["kategori"] == kategori]
    return cocok or list(range(len(SOAL)))


def tampilkan(i: int):
    i = max(0, min(int(i), len(SOAL) - 1))
    x = SOAL[i]
    opsi = [(f"{h}. {t}", h) for h, t in sorted(x["opsi"].items())]
    return i + 1, soal_html(x, i), gr.update(choices=opsi, value=None), ""


def ke_nomor(nomor: float):
    """Lompat ke satu soal tertentu. Nomornya sama dengan urutan berkas soal."""
    return tampilkan(int(nomor or 1) - 1)


def melangkah(kategori: str, nomor: float, arah: int):
    daftar = kandidat(kategori)
    i = max(0, min(int(nomor or 1) - 1, len(SOAL) - 1))
    if i in daftar:
        pos = daftar.index(i)
    else:
        # Nomor sedang berada di luar kategori yang dipilih. Yang dituju adalah
        # tetangga terdekat ke arah langkah, bukan awal daftar.
        pos = sum(1 for j in daftar if j < i) - (1 if arah > 0 else 0)
    return tampilkan(daftar[(pos + arah) % len(daftar)])


def acak(kategori: str, nomor: float):
    daftar = kandidat(kategori)
    kini = int(nomor or 1) - 1
    return tampilkan(random.choice([i for i in daftar if i != kini] or daftar))


def ke_kategori(kategori: str):
    return tampilkan(kandidat(kategori)[0])


def periksa(nomor: float, huruf: str) -> str:
    x = SOAL[max(0, min(int(nomor or 1) - 1, len(SOAL) - 1))]
    kunci = x["kunci"]
    benar = f"{kunci}. {x['opsi'][kunci]}"
    if not huruf:
        return ('<div class="lq-nilai lq-netral">Pilih salah satu opsi lebih '
                'dahulu.</div>')
    if huruf == kunci:
        return (f'<div class="lq-nilai lq-benar"><b>Benar.</b> Kuncinya '
                f'{html.escape(benar)}.</div>')
    return (f'<div class="lq-nilai lq-salah"><b>Belum tepat.</b> Kuncinya '
            f'{html.escape(benar)}.</div>')


# --- tabel hasil -------------------------------------------------------------

# Urutannya disengaja: pilihan ganda adalah tolok ukur utama, jadi ia yang
# terbuka lebih dulu. Mengurutkan menurut nama berkas justru menaruh jalur
# kedua di depan.
JUDUL_TABEL = {
    "mc--papan-skor.md": "Papan skor pilihan ganda (tolok ukur utama)",
    "full--papan-skor.md": "Papan skor teks bebas",
}
ADA = {p.name for p in (DATA / "tabel").glob("*.md")}
BERKAS = {judul: nama for nama, judul in JUDUL_TABEL.items() if nama in ADA}
TABEL = list(BERKAS) or ["(belum tersedia)"]

SKOR = json.loads((DATA / "skor.json").read_text(encoding="utf-8"))
METRIK = SKOR["metrik"]
LANTAI = SKOR["lantai"]
JALUR = {"Pilihan ganda": "mc.", "Teks bebas": "full.",
         "Benchmark luar": "eksternal."}
NAMA_JENIS = {"dilatih": "Hasil fine-tuning LaguQA",
              "dasar": "Model tanpa pelatihan",
              "kontrol": "Kontrol (tidak mengenal satu lagu pun)"}


def metrik_jalur(jalur: str) -> list[str]:
    return [k for k in METRIK if k.startswith(JALUR[jalur])]


def pendek(nama: str) -> str:
    """Nama untuk label diagram: awalan model yang sama bagi semua varian dibuang."""
    return nama.replace("gemma4-e2b [", "[").replace("gemma4-e2b", "gemma4-e2b (base)")


def catatan_tabel(nama: str) -> str:
    """Baris penjelas di bawah tabel markdown, termasuk sha256 berkas soalnya.

    Tabelnya sendiri digambar ulang di sini supaya bisa diberi warna, tetapi
    keterangan asalnya tetap diambil dari berkas yang sama, bukan diketik ulang.
    """
    p = DATA / "tabel" / BERKAS.get(nama, "")
    if not p.is_file():
        return ""
    baris = [b for b in p.read_text(encoding="utf-8").splitlines()
             if not b.startswith("|")]
    return "\n".join(baris).strip()


def sel(nilai: float | None, terbaik: bool, bawah: bool) -> str:
    if nilai is None:
        return '<td class="lq-angka lq-hampa">—</td>'
    kelas = "lq-angka" + (" lq-terbaik" if terbaik else "")
    tanda = ' <span class="lq-bawah" title="di bawah batas bawah">↓</span>' if bawah else ""
    return f'<td class="{kelas}">{nilai:.1f}{tanda}</td>'


def papan(jalur: str, model: list[dict], kelompokkan: bool = True) -> str:
    """Tabel skor berwarna: nilai terbaik tiap kolom disorot.

    Yang disorot dihitung di antara model saja, tanpa baris kontrol. Kontrol
    memenangkan beberapa kolom tanpa mengenal satu lagu pun -- pada abstain
    teks bebas ia mencetak 100 karena tidak pernah menjawab -- dan menyorotnya
    akan menobatkan penebak sebagai model terbaik.
    """
    kolom = metrik_jalur(jalur)
    if not kolom:
        return '<div class="lq-catatan">Belum ada angka untuk jalur ini.</div>'
    # Model yang belum diukur pada jalur ini dibuang, bukan ditampilkan dengan
    # tanda pisah sebaris penuh. Papan teks bebas baru memuat empat model, dan
    # dua puluh baris kosong di bawahnya membuat yang terukur sulit dicari.
    punya = [m for m in model if any(k in m["skor"] for k in kolom)]
    if not punya:
        return '<div class="lq-catatan">Belum ada model yang diukur di jalur ini.</div>'

    terbaik = {}
    for k in kolom:
        nilai = [m["skor"][k] for m in punya
                 if k in m["skor"] and m["jenis"] != "kontrol"]
        if nilai:
            terbaik[k] = max(nilai)

    kepala = "".join(f'<th class="lq-angka">{html.escape(METRIK[k]["label"].split(": ")[-1])}</th>'
                     for k in kolom)
    baris = [f'<thead><tr><th>Model</th>{kepala}</tr></thead><tbody>']
    jenis_sekarang = None
    for m in punya:
        if kelompokkan and m["jenis"] != jenis_sekarang:
            jenis_sekarang = m["jenis"]
            baris.append(f'<tr class="lq-grup"><td colspan="{len(kolom) + 1}">'
                         f'{html.escape(NAMA_JENIS.get(jenis_sekarang, jenis_sekarang))}'
                         f'</td></tr>')
        sel_baris = "".join(
            sel(m["skor"].get(k), m["skor"].get(k) == terbaik.get(k),
                k in LANTAI and m["jenis"] != "kontrol"
                and m["skor"].get(k, 0) < LANTAI[k])
            for k in kolom)
        gpu = f' <span class="lq-gpu">{m["gpu"]}</span>' if m.get("gpu") else ""
        baris.append(f'<tr><td class="lq-nama">{html.escape(m["nama"])}{gpu}</td>'
                     f'{sel_baris}</tr>')
    baris.append("</tbody>")
    return f'<div class="lq-gulir"><table class="lq-papan">{"".join(baris)}</table></div>'


def tabel(nama: str) -> tuple[str, str]:
    jalur = "Teks bebas" if nama.startswith("Papan skor teks") else "Pilihan ganda"
    return papan(jalur, SKOR["model"]), catatan_tabel(nama)


# --- diagram sebar -----------------------------------------------------------

WARNA = {"dilatih": "var(--primary-600)", "dasar": "var(--body-text-color)",
         "kontrol": "var(--body-text-color-subdued)"}


def batas(nilai: list[float], garis: float | None) -> tuple[float, float]:
    lo, hi = min(nilai), max(nilai)
    if garis is not None:
        lo, hi = min(lo, garis), max(hi, garis)
    if hi - lo < 1e-9:
        return lo - 1, hi + 1
    tepi = (hi - lo) * 0.12
    return lo - tepi, hi + tepi


def sebar(x_key: str, y_key: str, jenis: list[str]) -> str:
    """Satu titik satu model, sumbunya dipilih dari metrik yang sudah diukur.

    Model yang belum punya salah satu dari kedua metrik tidak digambar dan
    jumlahnya disebutkan. Menggambarnya di angka nol akan membuat model yang
    belum diukur tampak seperti model yang gagal.
    """
    if x_key == y_key:
        return '<div class="lq-catatan">Pilih dua metrik yang berbeda.</div>'
    ikut = [m for m in SKOR["model"] if m["jenis"] in (jenis or [])]
    titik = [m for m in ikut if x_key in m["skor"] and y_key in m["skor"]]
    lewat = len(ikut) - len(titik)
    if len(titik) < 2:
        return ('<div class="lq-catatan">Belum cukup model yang diukur pada '
                'kedua metrik itu. Coba metrik lain, atau tambahkan kelompok '
                'model.</div>')

    W, H = 760, 470
    kiri, kanan, atas, bawah = 62, 16, 18, 48
    lx = LANTAI.get(x_key)
    ly = LANTAI.get(y_key)
    x0, x1 = batas([m["skor"][x_key] for m in titik], lx)
    y0, y1 = batas([m["skor"][y_key] for m in titik], ly)
    px = lambda v: kiri + (v - x0) / (x1 - x0) * (W - kiri - kanan)
    py = lambda v: H - bawah - (v - y0) / (y1 - y0) * (H - atas - bawah)

    bagian = [f'<svg viewBox="0 0 {W} {H}" class="lq-sebar" '
              f'role="img" aria-label="diagram sebar perbandingan model">']

    def sumbu(lo, hi, buat):
        langkah = (hi - lo) / 4
        skala = 10 ** math.floor(math.log10(langkah)) if langkah > 0 else 1
        for m in (1, 2, 2.5, 5, 10):
            if skala * m >= langkah:
                langkah = skala * m
                break
        v = math.ceil(lo / langkah) * langkah
        while v <= hi + 1e-9:
            buat(v)
            v += langkah

    sumbu(y0, y1, lambda v: bagian.append(
        f'<line class="lq-kisi" x1="{kiri}" y1="{py(v):.1f}" '
        f'x2="{W - kanan}" y2="{py(v):.1f}"/>'
        f'<text class="lq-tik" x="{kiri - 8}" y="{py(v) + 3.5:.1f}" '
        f'text-anchor="end">{v:.0f}</text>'))
    sumbu(x0, x1, lambda v: bagian.append(
        f'<line class="lq-kisi" x1="{px(v):.1f}" y1="{atas}" '
        f'x2="{px(v):.1f}" y2="{H - bawah}"/>'
        f'<text class="lq-tik" x="{px(v):.1f}" y="{H - bawah + 18}" '
        f'text-anchor="middle">{v:.0f}</text>'))

    for nilai, arah in ((lx, "x"), (ly, "y")):
        if nilai is None:
            continue
        if arah == "x":
            bagian.append(f'<line class="lq-lantai" x1="{px(nilai):.1f}" '
                          f'y1="{atas}" x2="{px(nilai):.1f}" y2="{H - bawah}"/>')
        else:
            bagian.append(f'<line class="lq-lantai" x1="{kiri}" '
                          f'y1="{py(nilai):.1f}" x2="{W - kanan}" '
                          f'y2="{py(nilai):.1f}"/>')

    # Label ditumpuk dengan jarak minimum supaya dua model berskor mirip tidak
    # saling menimpa. Dua arah, bukan satu: mendorong ke bawah saja membuat
    # gerombolan di dasar grafik mendorong label terakhir keluar dari gambar,
    # dan label yang keluar bidang sama saja dengan hilang. Dorongan ke bawah
    # dijalankan lebih dulu, lalu apa pun yang melewati dasar didorong balik ke
    # atas.
    JARAK = 11
    urut = sorted(titik, key=lambda m: m["skor"][y_key], reverse=True)
    ty = []
    for m in urut:
        alami = py(m["skor"][y_key]) + 3.5
        ty.append(max(alami, ty[-1] + JARAK) if ty else alami)
    for i in range(len(ty) - 1, -1, -1):
        batas_bawah = H - bawah if i == len(ty) - 1 else ty[i + 1] - JARAK
        ty[i] = min(ty[i], batas_bawah)
    for m, label_y in zip(urut, ty):
        cx, cy = px(m["skor"][x_key]), py(m["skor"][y_key])
        ke_kiri = cx > W - kanan - 150
        tx = cx - 9 if ke_kiri else cx + 9
        anchor = "end" if ke_kiri else "start"
        warna = WARNA.get(m["jenis"], "var(--body-text-color)")
        bagian.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{warna}" '
            f'fill-opacity="{0.9 if m["jenis"] == "dilatih" else 0.55}" '
            f'stroke="{warna}"><title>{html.escape(m["nama"])}: '
            f'{m["skor"][x_key]:.1f} / {m["skor"][y_key]:.1f}</title></circle>'
            f'<text class="lq-label" x="{tx:.1f}" y="{label_y:.1f}" '
            f'text-anchor="{anchor}">{html.escape(pendek(m["nama"]))}</text>')

    bagian.append(f'<text class="lq-sumbu" x="{(kiri + W - kanan) / 2:.0f}" '
                  f'y="{H - 8}" text-anchor="middle">'
                  f'{html.escape(METRIK[x_key]["label"])}</text>')
    bagian.append(f'<text class="lq-sumbu" transform="translate(14,'
                  f'{(atas + H - bawah) / 2:.0f}) rotate(-90)" '
                  f'text-anchor="middle">{html.escape(METRIK[y_key]["label"])}'
                  f'</text>')
    bagian.append("</svg>")

    sisa = (f'<p>{lewat} model tidak digambar karena belum diukur pada salah '
            f'satu metrik itu.</p>' if lewat else "")
    return (f'<div class="lq-gulir">{"".join(bagian)}</div>'
            f'<div class="lq-catatan"><p>Garis putus-putus menandai batas '
            f'bawah metrik yang bersangkutan: skor yang dicapai tanpa mengenal '
            f'satu lagu pun.</p>{sisa}</div>')


# --- tampilan ----------------------------------------------------------------

# abcjs menggambar not balok dari sumber ABC yang sama dengan yang menjadi
# kunci jawaban soal notasi, lalu membunyikannya memakai soundfont FluidR3_GM.
# Keduanya berjalan di peramban, jadi ZeroGPU tidak terpakai untuk memutar
# lagu. Versinya dipatok supaya tampilan yang dilihat penguji hari ini sama
# dengan yang dilihat pembaca setahun lagi.
HEAD = f"""
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/abcjs@{ABCJS}/abcjs-audio.css">
<script src="https://cdn.jsdelivr.net/npm/abcjs@{ABCJS}/dist/abcjs-basic-min.js"></script>
<script>
window.laguqaGiliran = 0;
window.laguqaAbc = "";
window.laguqaJangkauan = "";

window.laguqaLolos = function (s) {{
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
                  .replace(/>/g, "&gt;");
}};

// Menandai jangkauan karakter pada tampilan sumber ABC. Isinya digambar ulang
// hanya ketika jangkauannya berubah; peristiwa abcjs berdatangan beberapa kali
// per detik dan menyusun ulang teks sepanjang itu setiap kali membuat
// penyorotannya tersendat.
window.laguqaSorotAbc = function (mulai, akhir) {{
  const el = document.getElementById("lq-abc-teks");
  if (!el) return;
  const kunci = mulai + ":" + akhir;
  if (kunci === window.laguqaJangkauan) return;
  window.laguqaJangkauan = kunci;
  const abc = window.laguqaAbc;
  const L = window.laguqaLolos;
  if (mulai === null || mulai === undefined || akhir === null
      || akhir === undefined || akhir <= mulai) {{
    el.innerHTML = L(abc);
    return;
  }}
  el.innerHTML = L(abc.slice(0, mulai))
    + "<mark>" + L(abc.slice(mulai, akhir)) + "</mark>"
    + L(abc.slice(akhir));
}};

window.laguqaTampil = async function (abc, alat) {{
  const kertas = document.getElementById("lq-kertas");
  if (!kertas || !window.ABCJS) return;

  // Penanda giliran. Menyiapkan suara menunggu unduhan soundfont, jadi dua
  // penggantian lagu yang berdekatan bisa selesai tidak berurutan dan yang
  // lama menimpa yang baru. Hanya panggilan terakhir yang boleh memasang
  // lagunya. Tanpa ini, mengganti lagu tetap memutar lagu sebelumnya.
  const giliran = ++window.laguqaGiliran;

  // Pemutarnya dibongkar, bukan sekadar dijeda. Menjeda lalu memanggil
  // setTune menyisakan penyangga suara lagu sebelumnya, sehingga tombol putar
  // masih membunyikan lagu yang lama meski partiturnya sudah berganti.
  if (window.laguqaSynth) {{
    try {{ window.laguqaSynth.destroy(); }} catch (e) {{}}
    window.laguqaSynth = null;
  }}
  const kosong = document.getElementById("lq-audio");
  if (kosong) kosong.innerHTML = "";

  window.laguqaAbc = abc || "";
  window.laguqaJangkauan = "";
  window.laguqaSorotAbc(null, null);
  if (!abc) {{ kertas.innerHTML = ""; return; }}

  const visual = ABCJS.renderAbc(kertas, abc, {{
    responsive: "resize",
    add_classes: true,
    staffwidth: 720,
    format: {{
      titlefont: "Inter 15 bold", subtitlefont: "Inter 12",
      composerfont: "Inter 11 italic", vocalfont: "Inter 11",
      gchordfont: "Inter 11", annotationfont: "Inter 10 italic",
      tempofont: "Inter 11"
    }}
  }})[0];

  const kotak = document.getElementById("lq-audio");
  if (!kotak) return;
  if (!ABCJS.synth.supportsAudio()) {{
    kotak.innerHTML = "<p class='lq-audio-galat'>Peramban ini tidak "
      + "mendukung pemutaran audio.</p>";
    return;
  }}

  {{
    // Kursor dan sorotan mengikuti not yang sedang berbunyi. Itu yang membuat
    // pratinjau ini berguna untuk memeriksa transkripsi: kalau not yang
    // disorot tidak cocok dengan yang terdengar, transkripsinya yang salah.
    const bersihkanSorotan = function () {{
      kertas.querySelectorAll(".abcjs-highlight").forEach(
        function (el) {{ el.classList.remove("abcjs-highlight"); }});
    }};
    const kursor = {{
      onStart: function () {{
        const svg = kertas.querySelector("svg");
        if (!svg || svg.querySelector(".abcjs-cursor")) return;
        const garis = document.createElementNS(
          "http://www.w3.org/2000/svg", "line");
        garis.setAttribute("class", "abcjs-cursor");
        ["x1", "y1", "x2", "y2"].forEach(function (a) {{
          garis.setAttributeNS(null, a, 0);
        }});
        svg.appendChild(garis);
      }},
      onEvent: function (ev) {{
        if (ev.measureStart && ev.left === null) return;

        const garis = kertas.querySelector(".abcjs-cursor");
        if (garis && ev.left !== undefined && ev.left !== null) {{
          const x = ev.left - 2;
          const y = ev.top || 0;
          const xLama = parseFloat(garis.getAttribute("x1") || "0");
          const yLama = parseFloat(garis.getAttribute("y1") || "0");
          // Meluncur halus antarnot pada baris yang sama, tetapi melompat
          // tanpa animasi ketika pindah baris atau mundur karena tanda ulang.
          // Tanpa pembedaan ini kursornya menyeberang layar secara diagonal.
          const lompat = Math.abs(y - yLama) > 5 || x < xLama;
          garis.style.transition = lompat ? "none"
            : "x1 .1s linear, x2 .1s linear, y1 .1s linear, y2 .1s linear";
          garis.setAttribute("x1", x);
          garis.setAttribute("x2", x);
          garis.setAttribute("y1", y);
          garis.setAttribute("y2", y + (ev.height || 0));
        }}

        bersihkanSorotan();
        (ev.elements || []).forEach(function (grup) {{
          (grup || []).forEach(function (el) {{
            el.classList.add("abcjs-highlight");
          }});
        }});

        // Bagian sumber ABC yang sedang berbunyi ikut ditandai. Peristiwa
        // abcjs membawa jangkauan karakternya, jadi penandaannya memakai
        // posisi yang sama dengan yang dipakai penggambar not.
        window.laguqaSorotAbc(ev.startChar, ev.endChar);
      }},
      onFinished: function () {{
        bersihkanSorotan();
        window.laguqaSorotAbc(null, null);
        const garis = kertas.querySelector(".abcjs-cursor");
        if (garis) {{
          garis.style.transition = "none";
          ["x1", "x2", "y1", "y2"].forEach(function (a) {{
            garis.setAttribute(a, 0);
          }});
        }}
      }}
    }};
    window.laguqaSynth = new ABCJS.synth.SynthController();
    window.laguqaSynth.load("#lq-audio", kursor, {{
      displayPlay: true, displayProgress: true, displayRestart: true,
      displayLoop: true, displayWarp: true, displayClock: true
    }});
  }}

  if (giliran !== window.laguqaGiliran) return;
  try {{
    await window.laguqaSynth.setTune(visual, false, {{
      program: parseInt(alat, 10) || 0,
      soundFontUrl:
        "https://paulrosen.github.io/midi-js-soundfonts/FluidR3_GM/"
    }});
    // Kalau lagunya sudah diganti lagi selama menunggu, biarkan panggilan
    // yang lebih baru yang menentukan apa yang terpasang.
    if (giliran === window.laguqaGiliran) {{
      window.laguqaSynth.setProgress(0, 1);
    }}
  }} catch (e) {{
    if (giliran === window.laguqaGiliran) {{
      kotak.innerHTML = "<p class='lq-audio-galat'>Suara gagal disiapkan: "
        + e + "</p>";
    }}
  }}
}};
</script>
"""

TEMA = gr.themes.Base(
    primary_hue=gr.themes.colors.red,
    secondary_hue=gr.themes.colors.red,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_sm,
    font=(gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui",
          "sans-serif"),
    font_mono=(gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace",
               "monospace"),
).set(
    # Rata tanpa bayangan dan tanpa gradien: seluruh pemisahan ruang dikerjakan
    # garis satu piksel dan jarak antarblok.
    shadow_drop="none", shadow_drop_lg="none", shadow_spread="0px",
    shadow_inset="none",
    block_shadow="none", block_label_shadow="none",
    block_border_width="1px", block_radius="6px",
    block_label_background_fill="transparent",
    block_label_text_weight="500",
    block_title_text_weight="500",
    input_shadow="none", input_shadow_focus="none", input_radius="6px",
    button_primary_shadow="none", button_primary_shadow_hover="none",
    button_primary_shadow_active="none",
    button_secondary_shadow="none", button_secondary_shadow_hover="none",
    button_secondary_shadow_active="none",
    checkbox_shadow="none", checkbox_label_shadow="none",
    button_large_radius="6px", button_small_radius="6px",
    button_medium_radius="6px",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_border_color="*primary_600",
    button_primary_border_color_hover="*primary_700",
    button_primary_text_color="#ffffff",
    button_primary_text_color_hover="#ffffff",
    button_secondary_background_fill="transparent",
    button_secondary_background_fill_hover="*neutral_100",
    button_secondary_border_color="*border_color_primary",
    checkbox_background_color_selected="*primary_600",
    checkbox_border_color_selected="*primary_600",
    slider_color="*primary_600",
)

CSS = """
/* margin auto ditulis eksplisit. max-width saja membuat isinya menempel ke
   tepi kiri pada layar lebar, karena Gradio 6 memakai lebar penuh sebagai
   bawaan dan tidak lagi menengahkan wadahnya sendiri. */
.gradio-container { max-width: 1080px !important;
                    margin-left: auto !important; margin-right: auto !important; }

/* --- kepala --- */
#lq-kepala { border-top: 3px solid var(--primary-600);
             border-bottom: 1px solid var(--border-color-primary);
             padding: 1.5rem 0 1.1rem; margin-bottom: .4rem; }
#lq-kepala .lq-merek { display: flex; align-items: center; gap: .9rem; }
/* Lambangnya angka 1 bergaris bawah, yaitu do seperdelapan dalam notasi
   angka. Garis bawah dipilih menggantikan titik oktaf karena angka bertitik
   di atas terbaca sebagai huruf i pada ukuran sekecil ini. */
#lq-kepala .lq-lambang { width: 44px; height: 44px; flex: 0 0 44px;
    background: var(--primary-600); color: #fff; border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--font-mono); font-size: 1.5rem; font-weight: 600;
    line-height: 1; position: relative; }
#lq-kepala .lq-lambang::after { content: ""; position: absolute; bottom: 11px;
    width: 15px; height: 2px; background: #fff; }
#lq-kepala h1 { font-size: 1.55rem; font-weight: 650; letter-spacing: -.02em;
                margin: 0; line-height: 1.15; }
#lq-kepala .lq-sub { color: var(--body-text-color-subdued); margin: .15rem 0 0;
                     font-size: .95rem; }
#lq-kepala .lq-angka { display: flex; flex-wrap: wrap; gap: .5rem;
                       margin: 1rem 0 0; }
#lq-kepala .lq-angka span { font-size: .82rem; padding: .25rem .6rem;
    border: 1px solid var(--border-color-primary); border-radius: 999px;
    color: var(--body-text-color-subdued); }
#lq-kepala .lq-angka b { color: var(--body-text-color); font-weight: 600; }

/* --- keping penanda --- */
.lq-keping { display: inline-block; font-size: .75rem; letter-spacing: .01em;
    padding: .16rem .5rem; border-radius: 4px; margin: 0 .35rem .35rem 0;
    border: 1px solid var(--border-color-primary);
    color: var(--body-text-color-subdued); }
.lq-keping.lq-jenis { border-color: var(--primary-600);
                      color: var(--primary-600); }
.lq-keping.lq-ok { border-color: var(--border-color-primary); }
.lq-keping.lq-awas { border-color: var(--primary-400);
                     color: var(--primary-600); }
.lq-keping.lq-kat { font-family: var(--font-mono); }

/* --- kartu lagu --- */
.lq-kartu { border: 1px solid var(--border-color-primary); border-radius: 6px;
            padding: 1rem 1.1rem; background: var(--background-fill-primary); }
.lq-kepala-kartu { display: flex; align-items: baseline; flex-wrap: wrap;
                   gap: .2rem .8rem; }
.lq-kartu h3 { margin: 0; font-size: 1.15rem; font-weight: 600;
               letter-spacing: -.01em; }
.lq-kartu dl { display: grid; grid-auto-flow: column;
    grid-template-rows: auto auto; justify-content: start;
    gap: .15rem 2.2rem; margin: .9rem 0 0; }
.lq-kartu dt { font-size: .76rem; color: var(--body-text-color-subdued); }
.lq-kartu dd { margin: 0; font-size: .95rem; }
.lq-kartu.lq-kosong { color: var(--body-text-color-subdued); }
.lq-catatan-kartu { margin: .9rem 0 0; padding-left: .8rem; font-size: .85rem;
    border-left: 2px solid var(--primary-500);
    color: var(--body-text-color-subdued); }
@media (max-width: 760px) {
  .lq-kartu dl { grid-auto-flow: row; grid-template-columns: 8rem 1fr;
                 gap: .35rem .8rem; }
}

/* --- not balok, digambar abcjs --- */
#lq-kertas { padding: .4rem .2rem; overflow-x: auto; }
#lq-kertas svg { max-width: 100%; }
#lq-kertas svg text, #lq-kertas svg tspan, #lq-kertas svg path {
    fill: var(--body-text-color); }
#lq-kertas svg path[stroke], #lq-kertas svg line, #lq-kertas svg .abcjs-staff,
#lq-kertas svg .abcjs-bar, #lq-kertas svg .abcjs-stem {
    stroke: var(--body-text-color); }
/* Sorotan ditulis dua kali: kepala not dan lirik memakai fill, sedangkan
   tangkai, balok, dan garis legar memakai stroke. Menyetel salah satunya saja
   membuat separuh not tetap berwarna teks biasa saat berbunyi. */
#lq-kertas svg .abcjs-highlight { fill: var(--primary-600); }
#lq-kertas svg path.abcjs-highlight[stroke],
#lq-kertas svg line.abcjs-highlight { stroke: var(--primary-600); }
#lq-kertas svg .abcjs-cursor { stroke: var(--primary-600); stroke-width: 2px;
    fill: none; opacity: .8; }
#lq-audio { margin-top: .5rem; }
#lq-audio .abcjs-inline-audio { background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary); border-radius: 6px;
    height: 48px; padding: 0 .5rem; box-shadow: none; gap: .15rem;
    color: var(--body-text-color); }
#lq-audio .abcjs-btn { background: none; border: 0; border-radius: 4px;
    width: 34px; height: 34px; padding: 6px; box-sizing: border-box; }
/* abcjs hanya memberi ukuran pada tombolnya, tidak pada svg di dalamnya, dan
   mengandalkan ukuran bawaan elemen svg. Gradio menimpa ukuran bawaan itu,
   sehingga ikonnya menjadi 0x0 dan tombolnya tampak hilang.
   Ukurannya saja yang disetel, bukan display: tombol putar memuat tiga ikon
   bertumpuk (putar, jeda, memuat) yang disembunyikan bergantian oleh abcjs,
   dan memaksa display:block di sini menampilkan ketiganya sekaligus. */
#lq-audio .abcjs-btn svg { width: 100%; height: 100%; }
#lq-audio .abcjs-midi-start .abcjs-pause-svg,
#lq-audio .abcjs-midi-start .abcjs-loading-svg,
#lq-audio .abcjs-midi-start.abcjs-pushed .abcjs-play-svg,
#lq-audio .abcjs-midi-start.abcjs-loading .abcjs-play-svg { display: none; }
#lq-audio .abcjs-midi-start.abcjs-pushed .abcjs-pause-svg { display: block; }
#lq-audio .abcjs-midi-start.abcjs-loading .abcjs-loading-svg {
    display: block; }
#lq-audio .abcjs-btn:hover { background: var(--background-fill-primary); }
#lq-audio .abcjs-btn g { fill: var(--body-text-color);
                         stroke: var(--body-text-color); }
#lq-audio .abcjs-btn:hover g { fill: var(--primary-600);
                               stroke: var(--primary-600); }
#lq-audio .abcjs-pushed { background: var(--background-fill-primary); }
#lq-audio .abcjs-midi-loop.abcjs-pushed g { fill: var(--primary-600);
                                            stroke: var(--primary-600); }
#lq-audio .abcjs-midi-progress-background {
    background: var(--background-fill-primary); height: 10px;
    border-radius: 5px; margin: 0 .7rem; flex: 1; cursor: pointer;
    border: 1px solid var(--border-color-primary); }
#lq-audio .abcjs-midi-progress-indicator { background: var(--primary-600);
    width: 14px; height: 14px; border-radius: 50%; margin-left: -7px;
    top: -3px; }
#lq-audio .abcjs-midi-clock { font-variant-numeric: tabular-nums;
                              font-size: .85rem; padding: 0 .5rem; }
#lq-audio .abcjs-midi-tempo { background: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary); border-radius: 4px;
    color: var(--body-text-color); width: 3.4rem; padding: .1rem .3rem; }
#lq-audio .abcjs-tempo-wrapper { font-size: .85rem;
                                 color: var(--body-text-color-subdued); }
.lq-audio-galat { font-size: .87rem; color: var(--body-text-color-subdued);
                  margin: 0; }

/* --- notasi angka --- */
.lq-notasi { display: flex; flex-wrap: wrap; font-family: var(--font-mono);
    font-size: .95rem; border: 1px solid var(--border-color-primary);
    border-radius: 6px; overflow: hidden;
    background: var(--background-fill-secondary); }
.lq-notasi span { padding: .55rem .7rem; letter-spacing: .06em;
                  border-right: 1px solid var(--border-color-primary); }
.lq-notasi span:last-child { border-right: 0; }
.lq-notasi.lq-kosong { display: block; padding: .8rem;
    font-family: var(--font); color: var(--body-text-color-subdued); }

/* --- soal --- */
.lq-soal { border: 1px solid var(--border-color-primary); border-radius: 6px;
           padding: 1rem 1.1rem; }
.lq-tanya { margin: .3rem 0 .8rem; font-size: 1.02rem; line-height: 1.55; }
.lq-kutipan { font-size: 1rem; line-height: 1.7; padding: .7rem .9rem;
    border-left: 2px solid var(--primary-600);
    background: var(--background-fill-secondary); }
.lq-nilai { padding: .65rem .9rem; border-radius: 6px; font-size: .92rem;
            border: 1px solid var(--border-color-primary); }
.lq-nilai.lq-benar { border-left: 3px solid var(--primary-600); }
.lq-nilai.lq-salah { border-left: 3px solid var(--border-color-primary); }
.lq-nilai.lq-netral { color: var(--body-text-color-subdued); }
.lq-peringatan { padding: .65rem .9rem; border-radius: 6px; font-size: .92rem;
    border: 1px solid var(--border-color-primary);
    border-left: 3px solid var(--primary-600); }

.lq-judul-kecil { font-size: .78rem; letter-spacing: .04em;
    text-transform: uppercase; color: var(--body-text-color-subdued);
    margin: 1.1rem 0 .35rem; }

/* --- sumber ABC --- */
#lq-abc-teks { font-family: var(--font-mono); font-size: .8rem;
    line-height: 1.55; white-space: pre-wrap; word-break: break-word;
    margin: 0; padding: .7rem .8rem; max-height: 260px; overflow: auto;
    border: 1px solid var(--border-color-primary); border-radius: 6px;
    background: var(--background-fill-secondary);
    color: var(--body-text-color); }
#lq-abc-teks mark { background: var(--primary-600); color: #fff;
                    border-radius: 2px; padding: 0 1px; }

/* --- papan skor --- */
.lq-gulir { overflow-x: auto; }
.lq-papan { border-collapse: collapse; width: 100%; font-size: .88rem; }
.lq-papan th, .lq-papan td { padding: .4rem .6rem; text-align: left;
    border-bottom: 1px solid var(--border-color-primary); white-space: nowrap; }
.lq-papan thead th { font-weight: 600; font-size: .8rem;
    color: var(--body-text-color-subdued); border-bottom-width: 2px; }
.lq-papan .lq-angka { text-align: right; font-variant-numeric: tabular-nums; }
.lq-papan .lq-nama { font-family: var(--font-mono); font-size: .84rem; }
.lq-papan tbody tr:hover td { background: var(--background-fill-secondary); }
.lq-papan .lq-grup td { font-weight: 600; font-size: .8rem; padding-top: .9rem;
    color: var(--body-text-color); border-bottom-color: transparent; }
/* Sorotan skor terbaik: latar tipis dan bukan teks tebal saja, supaya kolom
   yang dimenangkan tetap terbaca saat tabelnya digulir menyamping. */
.lq-papan td.lq-terbaik { background: var(--primary-600); color: #fff;
    font-weight: 600; border-radius: 3px; }
.lq-papan td.lq-hampa { color: var(--body-text-color-subdued); }
.lq-papan .lq-bawah { color: var(--primary-400); }
.lq-papan .lq-gpu { font-size: .7rem; margin-left: .35rem; padding: 0 .3rem;
    border: 1px solid var(--border-color-primary); border-radius: 3px;
    color: var(--body-text-color-subdued); }

/* --- diagram sebar --- */
.lq-sebar { width: 100%; min-width: 620px; height: auto; }
.lq-sebar .lq-kisi { stroke: var(--border-color-primary); stroke-width: 1; }
.lq-sebar .lq-lantai { stroke: var(--primary-500); stroke-width: 1.5;
    stroke-dasharray: 5 4; opacity: .75; }
.lq-sebar .lq-tik { font-size: 11px; fill: var(--body-text-color-subdued); }
.lq-sebar .lq-label { font-size: 10.5px; fill: var(--body-text-color-subdued); }
.lq-sebar .lq-sumbu { font-size: 12px; fill: var(--body-text-color);
    font-weight: 600; }

/* --- catatan kaki tiap tab --- */
.lq-catatan { font-size: .87rem; color: var(--body-text-color-subdued);
    border-top: 1px solid var(--border-color-primary);
    padding-top: .9rem; margin-top: 1.4rem; }
.lq-catatan p { margin: .45rem 0; }
"""

KEPALA = f"""
<div id="lq-kepala">
  <div class="lq-merek">
    <div class="lq-lambang">1</div>
    <div>
      <h1>LaguQA</h1>
      <p class="lq-sub">Sejauh mana model bahasa mengenal lagu nasional dan
      lagu daerah Indonesia</p>
    </div>
  </div>
  <div class="lq-angka">
    <span><b>{ribuan(RINGKAS['lagu'])}</b> lagu</span>
    <span><b>{ribuan(RINGKAS['soal'])}</b> soal pilihan ganda</span>
    <span><b>{RINGKAS['kategori']}</b> kategori</span>
    <span><b>{RINGKAS['pembanding']}</b> model pembanding</span>
    <span><b>{RINGKAS['terverifikasi']}</b> notasi terverifikasi</span>
  </div>
</div>
"""

with gr.Blocks(title="LaguQA", analytics_enabled=False) as demo:
    gr.HTML(KEPALA)

    with gr.Tab("Percakapan"):
        if model is None:
            gr.HTML(peringatan_model())
        gr.ChatInterface(
            fn=jawab,
            chatbot=gr.Chatbot(label="Percakapan", height=420),
            textbox=gr.Textbox(placeholder="Tanyakan sesuatu tentang lagunya",
                               label="Pertanyaan", submit_btn="Kirim"),
            additional_inputs=[
                gr.Slider(0, 1.2, value=0.0, step=0.1, label="Temperature"),
                gr.Slider(64, 512, value=192, step=64,
                          label="Panjang jawaban maksimum (token)"),
            ],
            additional_inputs_accordion=gr.Accordion("Pengaturan", open=False),
            examples=[["Siapa pencipta lagu Gugur Bunga?"],
                      ["Lagu Bungong Jeumpa berasal dari daerah mana?"],
                      ["Apa nada dasar lagu Indonesia Raya?"],
                      ["Tuliskan notasi angka bar 1 sampai 4 lagu Syukur."]],
            cache_examples=False,
        )
        gr.HTML(
            f'<div class="lq-catatan">'
            f'<p>Model <code>{html.escape(ADAPTER_ID)}</code> dijalankan di '
            f'atas <code>{html.escape(BASE_ID)}</code>.</p>'
            f'<p>Temperature 0 memberi jawaban yang sama setiap kali '
            f'dijalankan, sama seperti pengaturan waktu pengujian. Nilai di '
            f'atasnya hanya diperlukan bila ingin melihat variasi jawaban.</p>'
            f'<p>Demonstrasi ini memakai ZeroGPU dengan kuota harian 2 menit '
            f'untuk pengunjung tanpa akun dan 5 menit untuk pengunjung '
            f'berakun, jadi jawaban yang panjang menghabiskannya lebih '
            f'cepat.</p></div>')

    with gr.Tab("Bandingkan jawaban"):
        gr.Markdown(
            "Satu pertanyaan, dijawab dua kali: sekali oleh Gemma yang belum "
            "dilatih, sekali oleh model hasil fine-tuning LaguQA. Keduanya "
            "menerima system prompt yang sama.")
        if model is None:
            gr.HTML(peringatan_model())
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"#### Tanpa pelatihan\n`{BASE_ID}`")
                chat_dasar = gr.Chatbot(label="", height=380)
            with gr.Column():
                gr.Markdown(f"#### Hasil fine-tuning LaguQA\n`{ADAPTER_ID}`")
                chat_latih = gr.Chatbot(label="", height=380)
        tanya = gr.Textbox(label="Pertanyaan", lines=1,
                           placeholder="Tanyakan sesuatu, keduanya menjawab sekaligus",
                           submit_btn=True)
        with gr.Row():
            kirim_b = gr.Button("Kirim ke keduanya", variant="primary")
            bersih_b = gr.Button("Bersihkan")
        with gr.Accordion("Pengaturan", open=False):
            suhu_b = gr.Slider(0, 1.2, value=0.0, step=0.1, label="Temperature")
            token_b = gr.Slider(64, 512, value=192, step=64,
                                label="Panjang jawaban maksimum (token)")
        # cache_examples wajib False, bukan dibiarkan None. Bawaannya di Spaces
        # menyala, dan menyimpan hasil contoh berarti Gradio menjalankan fungsi
        # berdekorator @spaces.GPU saat aplikasi start, ketika belum ada GPU
        # yang dialokasikan. Gagalnya berupa "No CUDA GPUs are available" dari
        # dalam worker ZeroGPU, jauh dari contohnya, sehingga terbaca seperti
        # kerusakan model.
        gr.Examples(
            examples=[["Siapa pencipta lagu Gugur Bunga?"],
                      ["Lagu Bungong Jeumpa berasal dari daerah mana?"],
                      ["Apa nada dasar lagu Indonesia Raya?"],
                      ["Tuliskan notasi angka bar 1 sampai 4 lagu Syukur."]],
            inputs=[tanya], cache_examples=False)
        for pemicu in (kirim_b.click, tanya.submit):
            pemicu(bandingkan, [tanya, chat_dasar, chat_latih, suhu_b, token_b],
                   [chat_dasar, chat_latih, tanya])
        bersih_b.click(lambda: ([], [], ""), None,
                       [chat_dasar, chat_latih, tanya])
        gr.HTML(
            '<div class="lq-catatan">'
            '<p>Dua percakapan berjalan bersamaan atas pertanyaan yang sama, '
            'dan masing-masing meneruskan riwayatnya sendiri. Setelah giliran '
            'pertama jawaban keduanya berbeda, jadi menyatukan riwayatnya akan '
            'meminta satu sisi melanjutkan kalimat yang tidak pernah ia '
            'ucapkan.</p>'
            '<p>Kedua jawaban keluar dari berkas bobot yang sama. Bobot LoRA '
            'dimatikan sementara untuk sisi kiri, jadi tidak ada model kedua '
            'yang diunduh dan tidak ada perbedaan versi di antara keduanya.</p>'
            '<p>Sekali kirim menghasilkan dua jawaban, sehingga kuota GPU-nya '
            'kira-kira dua kali percakapan biasa. Temperature 0 membuat '
            'keduanya dapat diulang persis.</p>'
            '<p>Perbandingan ini memperlihatkan selisih pada satu percakapan, '
            'bukan mengukurnya. Angka yang terukur ada di tab Hasil, dihitung '
            'atas 1.200 soal.</p></div>')

    with gr.Tab("Lagu") as tab_lagu:
        with gr.Row():
            cari = gr.Textbox(label="Cari", scale=2,
                              placeholder="judul, pencipta, atau daerah asal")
            jenis = gr.Radio(["Semua", "Nasional", "Daerah"], value="Semua",
                             label="Jenis", scale=2)
            urutan = gr.Radio([URUT_BUKU, URUT_ABJAD], value=URUT_BUKU,
                              label="Urutan", scale=2)
        with gr.Row():
            pilih = gr.Dropdown(NAMA, value=NAMA[0], label="Lagu", scale=4,
                                filterable=True)
            alat = gr.Dropdown([(k, str(v)) for k, v in ALAT.items()],
                               value="0", label="Alat musik", scale=2)
        kartu = gr.HTML()
        # Kotak ABC tidak ditampilkan; keberadaannya semata supaya nilai teks
        # ABC ikut terkirim ke penangan JavaScript sebagai masukan biasa.
        abc = gr.Textbox(visible=False)
        gr.HTML('<div id="lq-kertas"></div><div id="lq-audio"></div>',
                padding=False)
        with gr.Row():
            with gr.Column():
                gr.HTML('<p class="lq-judul-kecil">Notasi angka, delapan bar '
                        'pertama</p>', padding=False)
                notasi = gr.HTML(padding=False)
            with gr.Column():
                gr.HTML('<p class="lq-judul-kecil">Sumber ABC</p>',
                        padding=False)
                gr.HTML('<pre id="lq-abc-teks"></pre>', padding=False)

        gambar = dict(fn=None, inputs=[abc, alat], outputs=None,
                      js="(abc, alat) => window.laguqaTampil(abc, alat)")
        atur = dict(fn=saring, inputs=[cari, jenis, urutan, pilih],
                    outputs=pilih)
        cari.change(**atur)
        jenis.change(**atur)
        urutan.change(**atur)
        pilih.change(kartu_lagu, pilih, [kartu, notasi, abc]).then(**gambar)
        alat.change(**gambar)
        demo.load(kartu_lagu, pilih, [kartu, notasi, abc])
        # Gradio baru membangun isi tab ketika tab itu dibuka, sehingga saat
        # demo.load berjalan elemen #lq-kertas belum ada di halaman. Menggambar
        # ulang saat tab dipilih adalah yang membuat not baloknya muncul pada
        # kunjungan pertama, bukan baru setelah lagunya diganti.
        tab_lagu.select(**gambar)

        gr.HTML(
            f'<div class="lq-catatan">'
            f'<p>Not balok digambar dan dibunyikan abcjs {ABCJS} langsung dari '
            f'berkas ABC hasil transkripsi, yaitu berkas yang sama dengan '
            f'yang menjadi kunci jawaban soal notasi. Yang terdengar adalah '
            f'transkripsinya, bukan rekaman lagunya, dan tanpa aransemen '
            f'bukunya.</p>'
            f'<p>Sebanyak {MENTAH} dari {RINGKAS["lagu"]} notasi masih '
            f'berstatus mentah karena belum lolos pemeriksaan konservasi '
            f'ketukan dan keselarasan lirik. Statusnya ditulis pada tiap '
            f'lagu, dan kekeliruannya biasanya terdengar.</p>'
            f'<p>Baris lirik ikut tergambar di bawah not karena tersimpan '
            f'pada berkas ABC-nya. Catatan hak untuk tiap lagu ada pada '
            f'berkas HAK-CIPTA.md di rilis datasetnya.</p></div>')

    with gr.Tab("Soal"):
        with gr.Row():
            kategori = gr.Dropdown([SEMUA] + KATEGORI, value=SEMUA,
                                   label="Kategori", scale=4)
            nomor = gr.Number(value=1, precision=0, minimum=1,
                              maximum=len(SOAL), step=1, label="Soal ke-",
                              scale=2)
            buka = gr.Button("Buka", scale=0)
            mundur = gr.Button("Sebelumnya", scale=0)
            maju = gr.Button("Berikutnya", scale=0)
            lagi = gr.Button("Acak", scale=0)
        pertanyaan = gr.HTML()
        opsi = gr.Radio([], label="Pilihan jawaban")
        jawabi = gr.Button("Periksa jawaban", variant="primary")
        nilai = gr.HTML()

        keluaran = [nomor, pertanyaan, opsi, nilai]
        kategori.change(ke_kategori, kategori, keluaran)
        # Tombol Buka disediakan karena Enter pada kotak angka tidak selalu
        # sampai ke peladen. Keduanya dipasang supaya cara mana pun berhasil.
        buka.click(ke_nomor, nomor, keluaran)
        nomor.submit(ke_nomor, nomor, keluaran)
        mundur.click(lambda k, n: melangkah(k, n, -1), [kategori, nomor],
                     keluaran)
        maju.click(lambda k, n: melangkah(k, n, 1), [kategori, nomor], keluaran)
        lagi.click(acak, [kategori, nomor], keluaran)
        jawabi.click(periksa, [nomor, opsi], nilai)
        demo.load(lambda: tampilkan(0), None, keluaran)

        gr.HTML(
            f'<div class="lq-catatan">'
            f'<p>Ditampilkan {len(SOAL)} soal contoh dari '
            f'{ribuan(RINGKAS["soal"])} soal, dengan lima opsi dan posisi '
            f'kunci yang diacak. Nomor soal dapat diketik langsung, dan '
            f'urutannya sama dengan urutan baris pada berkas soalnya, '
            f'sehingga tiap soal di sini dapat dicocokkan dengan '
            f'sumbernya.</p>'
            f'<p>Model tidak dinilai dari huruf yang ditulisnya, melainkan '
            f'dari log-probability teks tiap opsi. Cara itu '
            f'menghindari dua kesalahan: model yang menjawab dengan kalimat '
            f'panjang dianggap salah, dan model yang gemar menulis huruf A '
            f'dianggap tahu.</p>'
            f'<p>Kategori <code>rumpang</code> dan <code>lirik_ke_judul</code> '
            f'menguji hafalan lirik, sedangkan <code>hitung_bar</code> dan '
            f'<code>nada_tertinggi</code> dapat dikerjakan dari notasi yang '
            f'tertera di soal tanpa mengenal lagunya.</p></div>')

    with gr.Tab("Hasil"):
        pilihan = gr.Dropdown(TABEL, value=TABEL[0], label="Tabel")
        isi = gr.HTML()
        sumber = gr.Markdown()
        pilihan.change(tabel, pilihan, [isi, sumber])
        demo.load(tabel, pilihan, [isi, sumber])
        gr.HTML(
            '<div class="lq-catatan">'
            '<p>Pembandingnya baris kontrol, bukan angka nol. Kunci soal '
            'birama 70,2 persen bernilai 4/4 dan kunci soal nada dasar 70,9 '
            'persen bernilai Do = C, sehingga penebak yang hafal sebaran itu '
            'dan tidak mengenal satu lagu pun sudah memperoleh 32,1 persen. '
            'Model di bawah angka tersebut tahu lebih sedikit tentang buku '
            'ini daripada penebak tadi.</p>'
            '<p>Setiap angka dihitung ulang dari berkas prediksi oleh program '
            'penilai yang sama dengan yang dipakai dalam penelitian. Tidak '
            'ada angka yang diketik manual, dan tiap tabel mencantumkan '
            'sha256 berkas soal yang dipakai.</p>'
            '<p>Kolom yang disorot adalah skor tertinggi di antara model. '
            'Baris kontrol tidak ikut diperebutkan: kontrol yang tidak pernah '
            'menjawab memenangkan kolom abstain tanpa mengenal satu lagu '
            'pun.</p></div>')

        gr.Markdown("### Diagram sebar")
        # Bawaannya sengaja pilihan ganda LaguQA lawan IndoMMLU. Itu satu-satunya
        # pasangan yang menjawab pertanyaan yang paling sering diajukan tentang
        # benchmark baru: apakah ia mengukur sesuatu yang benchmark yang sudah
        # ada belum mengukur.
        SUMBU_X = "eksternal.indommlu" if "eksternal.indommlu" in METRIK \
            else "full.Keseluruhan"
        with gr.Row():
            x_pilih = gr.Dropdown([(v["label"], k) for k, v in METRIK.items()],
                                  value=SUMBU_X, label="Sumbu datar")
            y_pilih = gr.Dropdown([(v["label"], k) for k, v in METRIK.items()],
                                  value="mc.Keseluruhan", label="Sumbu tegak")
        jenis_pilih = gr.CheckboxGroup(
            [(NAMA_JENIS[j], j) for j in ("dilatih", "dasar", "kontrol")],
            value=["dilatih", "dasar"], label="Kelompok yang digambar")
        gambar = gr.HTML()
        for pemicu in (x_pilih.change, y_pilih.change, jenis_pilih.change):
            pemicu(sebar, [x_pilih, y_pilih, jenis_pilih], gambar)
        demo.load(sebar, [x_pilih, y_pilih, jenis_pilih], gambar)

        gr.HTML(
            '<div class="lq-catatan">'
            '<p>Angka IndoMMLU dan IndoCulture diukur dengan penilai yang sama '
            'dengan LaguQA, memakai system prompt netral. Kondisi prompt yang '
            'menyebut lagu juga diukur dan selisihnya di bawah 1,5 poin, '
            'sehingga urutannya tidak berubah.</p>'
            '<p>Penanda L4 dan L40S menyebut GPU tempat model diukur. Bobot, '
            'berkas, dan kode yang sama pada dua GPU itu berbeda jawaban pada '
            '1,5 sampai 2 persen soal, tetapi akurasinya hanya berbeda 0,0 '
            'sampai 0,4 poin.</p></div>')

if __name__ == "__main__":
    # Sejak Gradio 6, tema, CSS, dan isi head diberikan di launch(),
    # bukan di Blocks.
    demo.launch(theme=TEMA, css=CSS, head=HEAD)
