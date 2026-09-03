# LaguQA

Satu buku kumpulan lagu, 107 lagu nasional dan lagu daerah Indonesia, seluruh
melodinya ditranskripsi tangan ke notasi ABC 2.1. Dari transkripsi itu
dibangkitkan 1.200 soal pilihan ganda berlima opsi, dan tiap kunci jawabannya
dapat ditelusuri ke halaman tertentu di buku tersebut.

Yang ditanyakan adalah apa yang tercetak di bukunya: pencipta, daerah asal, nada
dasar, birama, istilah tempo, lirik, dan melodinya. Sebagian soal menyodorkan
potongan notasi angka lalu menanyakan lagunya. Sebagian lagi menanyakan jumlah
bar atau nada tertinggi. Tidak ada berkas audio yang dipakai.

Notasi angka dipakai untuk mengajarkan musik di sebagian besar sekolah di
Indonesia. Belum ada *benchmark* yang menguji apakah model bahasa bisa
membacanya.

Bagian dari penelitian skripsi Program Studi Informatika, Universitas Ahmad
Dahlan.

[Read this in English](README.md)

| | |
|---|---|
| Situs | [laguqa.github.io](https://laguqa.github.io) |
| Kode | [github.com/IRedDragonICY/LaguQA](https://github.com/IRedDragonICY/LaguQA) |
| Demo | [IRedDragonICY/LaguQA-Demo](https://huggingface.co/spaces/IRedDragonICY/LaguQA-Demo) |
| Dataset | [IRedDragonICY/LaguQA](https://huggingface.co/datasets/IRedDragonICY/LaguQA) |
| Cermin dataset | [kaggle.com/datasets/ireddragonicy/laguqa](https://www.kaggle.com/datasets/ireddragonicy/laguqa) |
| Model | [IRedDragonICY/LaguQA-Gemma4-E2B](https://huggingface.co/IRedDragonICY/LaguQA-Gemma4-E2B) |
| Semuanya di Hugging Face | [Koleksi LaguQA](https://huggingface.co/collections/IRedDragonICY/laguqa-6a9826460786a111107e430e) |

## Hasil

Tiga belas model dinilai tanpa pelatihan pada seluruh 1.200 soal. Tidak satu pun
melampaui penebak yang tidak mengenal satu lagu pun di dalamnya.

| | Akurasi (%) |
|---|---|
| Tebakan yang mengikuti sebaran kunci | 32,1 |
| Model tanpa pelatihan terbaik, sahabatai-9b | 29,5 |
| gemma4-e2b tanpa pelatihan | 24,1 |
| gemma4-e2b sesudah LoRA, tiga *seed* | 52,3 sampai 61,0 |
| Tebakan acak | 17,2 |

Batas bawah di sini penebak yang mengikuti sebaran kunci, bukan tebakan acak.
Kunci soal birama 70,2% bernilai 4/4 dan kunci soal nada dasar 70,9% bernilai
Do = C. Hafalkan dua hal itu, tanpa tahu apa pun selebihnya, hasilnya sudah
32,1%. Tebakan acak hanya 17,2%, dan memakainya sebagai batas bawah akan membuat
setiap model tampak berpengetahuan.

Pelatihan mengangkat gemma4-e2b ke rentang lima puluhan sampai enam puluhan.
Ketiga angka itu berasal dari satu resep yang dijalankan tiga kali dengan *seed*
berbeda, tanpa ada lagi yang diubah: berkas latih ber-sha256 sama, *learning
rate* sama, 2.622 langkah yang sama. Selisihnya 8,7 poin. Angka itu lebih lebar
daripada kebanyakan selisih antarsetelan yang biasa dilaporkan orang, sehingga
satu percobaan tunggal tidak membuktikan banyak hal.

Skor tiap kategori untuk seluruh model ada di
[`docs/tabel/mc/papan-skor.md`](docs/tabel/mc/papan-skor.md).

## Isi repositori ini

| Folder | Isi |
|---|---|
| `src/laguqa/scans/` | penyiapan pindaian halaman buku |
| `src/laguqa/notation/` | pemeriksa dan pengubah notasi ABC 2.1 |
| `src/laguqa/dataset/` | perakit tabel lagu |
| `src/laguqa/benchmark/` | pembangkit soal, penilai, dan kontrol |
| `src/laguqa/report/` | papan skor, tabel, dan grafik hasil |
| `scripts/` | jalur pipa, bernomor menurut urutan jalannya |
| `hasil/` | manifes dan jawaban tiap butir untuk tiap percobaan |
| `tests/` | uji otomatis |
| `modal_train.py` | pelatihan LoRA di Modal |

Tiga hal terbit di tempat lain. Data lagunya turunan buku bercopyright dan
terbit dengan lisensinya sendiri lewat Hugging Face dan Kaggle. Bobot
adapternya ratusan megabyte per percobaan dan terbit lewat Hugging Face. Naskah
skripsi beserta program penyusunnya bukan kode penelitian dan sama sekali tidak
masuk kontrol versi.

## Menjalankan ulang

```bash
pip install -e .
```

Jalur notasi dan pembangkit soal sengaja ditulis tanpa pustaka di luar pustaka
standar, supaya *benchmark* ini dapat dibangun ulang tanpa perkakas kompilasi.
Pelatihan dan penilaian tambahan memerlukan `modal`, `matplotlib`, dan `pandas`.

Ambil datanya lebih dulu:

```bash
hf download IRedDragonICY/LaguQA --repo-type dataset --local-dir data
```

Membangkitkan ulang soal dari tabel lagu:

```bash
python scripts/10_generate_benchmark.py
```

Menilai satu model pada jalur pilihan ganda:

```bash
python scripts/22_evaluate_mc.py
```

Melatih adapter LoRA:

```bash
python modal_train.py
```

Menyusun ulang papan skor dan grafiknya dari isi `hasil/`:

```bash
python scripts/19_leaderboard.py && python scripts/31_charts.py
```

Skrip dijalankan menurut urutan nomornya, dari penyiapan pindaian sampai
penerbitan rilis. Tiap skrip menerima `--help`.

## Cara jawaban dinilai

Model tidak pernah diminta mengetik huruf jawabannya. Untuk tiap soal, teks
kelima opsi disambungkan ke *prompt* satu per satu, dihitung rerata
*log-probability* token opsi tersebut, lalu nilai tertinggi dianggap jawabannya.

Dua cara lain sempat dicoba lebih dulu dan keduanya menyesatkan. Menilai dari
teks yang dibangkitkan menghukum model yang menjawab panjang: dua model
pembanding menulis penalaran terbuka sampai kehabisan token tanpa pernah
memutuskan satu opsi. Menilai dari peluang satu token huruf mengukur huruf mana
yang disukai model, dan itu ternyata sedikit sekali kaitannya dengan apakah
model mengenal lagunya.

## Lisensi

Lisensi program ini belum ditetapkan, sehingga hak ciptanya masih penuh pada
penulis. Datanya terbit terpisah dengan lisensi CC BY-NC 4.0 karena turunan buku
cetak; keterangan hak ciptanya menyertai berkas datasetnya.

## Sitasi

```bibtex
@misc{hendianto2026laguqa,
  author       = {Hendianto, Mohammad Farid},
  title        = {{LaguQA}: A Benchmark for Indonesian National and Regional
                  Song Understanding in Large Language Models},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/datasets/IRedDragonICY/LaguQA}},
  note         = {Skripsi, Program Studi Informatika,
                  Universitas Ahmad Dahlan}
}
```
