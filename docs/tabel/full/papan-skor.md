Jalur teks bebas, akurasi toleran (%). Dinilai `scripts/11_evaluate.py`.

| Model | Keseluruhan | Fakta | Abstain | Verifikasi | Lirik | Notasi | Penalaran |
|---|---|---|---|---|---|---|---|
| **Kontrol (tidak mengenal satu lagu pun)** | | | | | | | |
| Tebakan tersering | 32.8 | 39.7 | 100.0 | 50.7 | 2.8 | 6.0 | 19.0 |
| Tebakan acak | 26.5 | 27.3 | 100.0 | 43.3 | 1.6 | 4.0 | 9.0 |
| Tidak menjawab | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **Model tanpa pelatihan** | | | | | | | |
| gemma4-e2b | 19.5 ↓ | 14.3 | 36.3 | 40.7 | 0.4 | 0.0 | 53.0 |
| **Hasil fine-tuning LaguQA** | | | | | | | |
| gemma4-e2b [penuh15] (seed 1) | 77.1 | 99.7 | 100.0 | 79.3 | 40.4 | 52.0 | 100.0 |
| gemma4-e2b [lr4e4] (seed 1) | 76.6 | 97.7 | 98.0 | 73.3 | 48.0 | 45.0 | 100.0 |
| gemma4-e2b [final] (seed 1) | 76.5 | 100.0 | 100.0 | 70.0 | 45.6 | 46.0 | 100.0 |

↓ menandai skor di bawah batas bawah, yang dipegang tebakan tersering pada 32,8%. Model bertanda itu tahu lebih sedikit tentang buku ini daripada penebak yang tidak mengenal satu lagu pun.

Kolom **Abstain** harus dibaca berpasangan dengan **Fakta**, tidak sendirian. Soal abstain menguji apakah model menolak mengarang ketika bukunya memang tidak mencantumkan apa-apa, sehingga model yang menolak menjawab segalanya mendapat 100% di sana, dan kedua kontrol memang begitu. Angka abstain yang rendah berarti model mengarang; angka abstain tinggi berarti sesuatu hanya jika kolom Fakta juga tinggi.

Dibangun dari laguqa_test.jsonl sha256 `3dee138f2dc2981e` pada 2026-09-03 05:35.
